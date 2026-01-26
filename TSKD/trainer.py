import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import copy
from tqdm import tqdm
import random
import os
import numpy as np
import psutil
import time
from torch.utils.data import DataLoader
from .model import CTRModel
from .framework import TSKD_Framework, evaluate_model, plot_loss_curve, create_separate_optimizers
from .dataset_stream import SplitHandle, StreamingReplayDataset, StreamingHDF5IndexDataset, StreamingReplayDataset, StreamingRowDataset
from .dataset import MyDataset, force_garbage_collection


def generate_model_info_string(config, model_type):
    """Generate a string containing model architecture and dimension information."""
    if model_type == 'teacher':
        arch = config['teacher_arch']
        embed_dim = config['teacher_embedding_dim']
        hidden_dims = '_'.join(map(str, config['hidden_dims_teacher']))
    elif model_type == 'student':
        arch = config['student_arch']
        embed_dim = config['student_embedding_dim']
        hidden_dims = '_'.join(map(str, config['hidden_dims_student']))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return f"{arch}_emb{embed_dim}_hidden{hidden_dims}"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def _log_memory(logger, tag: str):
    """Log process RSS and GPU VRAM usage with a tag."""
    proc = psutil.Process(os.getpid())
    rss_gb = proc.memory_info().rss / (1024 ** 3)
    msg = f"[MEM] {tag} | RSS: {rss_gb:.2f} GB"
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        # Prefer mem_get_info if available
        if hasattr(torch.cuda, 'mem_get_info'):
            free_b, total_b = torch.cuda.mem_get_info(device)
            used_gb = (total_b - free_b) / (1024 ** 3)
            msg += f", GPU Used: {used_gb:.2f} GB / {total_gb:.2f} GB"
        else:
            alloc_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
            msg += f", GPU Alloc: {alloc_gb:.2f} GB, Reserved: {reserved_gb:.2f} GB / {total_gb:.2f} GB"
    logger.info(msg)


def _append_to_result_file(line: str, filename: str = "result.txt") -> None:
    """Append a single line to the result file (UTF-8). Silently ignore I/O errors."""
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except Exception:
        # Intentionally no logging to avoid recursion when logger depends on file handlers
        pass


def create_dataloader(config, dataset, name, num_cols, cat_cols, vocab, logger):
    """
    创建所有需要的dataloader，避免重复创建
    
    Args:
        config: 配置字典
        df_hist, df_offline, df_online, df_test: 数据DataFrame
        num_cols, cat_cols: 数值和类别特征列
        vocab: 词汇表
        logger: 日志记录器
    
    Returns:
        dict: 包含所有dataloader的字典
    """
    if config.get('dataset') == 'criteo_ctr_full' and isinstance(dataset, SplitHandle):
        # 仅当传入的是 SplitHandle（流式）时，才使用流式 DataLoader
        stream_workers = config['stream_num_workers']
        if stream_workers > 0:
            ds = StreamingRowDataset(dataset, page_size=config['stream_page_size'])
            logger.info(f"Creating {name} (row-stream) loader with {stream_workers} workers (page cache)")
            return DataLoader(
                ds,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=stream_workers,
                pin_memory=config['pin_memory'],
                prefetch_factor=config['prefetch_factor'],
                persistent_workers=config['persistent_workers']
            )
        else:
            ds = StreamingHDF5IndexDataset(dataset, cache_blocks=config['stream_cache_blocks'])
            logger.info(f"Creating {name} (stream) loader with 0 workers (HDF5/Parquet streaming)")
            return DataLoader(
                ds,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=stream_workers,
                collate_fn=ds.collate_fn,
                pin_memory=config['pin_memory']
            )
    else:
        logger.info(f"Creating {name} loader with {config['num_workers']} workers")
        return DataLoader(
            MyDataset(dataset, num_cols, cat_cols, vocab, config['dataset']),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            prefetch_factor=config['prefetch_factor'],
            persistent_workers=config['persistent_workers'],
        )

def train(config, device, df_hist, df_offline, df_online, df_test, num_cols, cat_cols, vocab, vocab_sizes, logger):   
    # Log command line arguments
    logger.info("="*80)
    logger.info("COMMAND LINE ARGUMENTS")
    logger.info("="*80)
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    logger.info("="*80)

    models_dir = config['dataset'] + "_saved_models"
    plot_dir = config['dataset'] + "_loss_curves"

    # Choose running mode
    if config['mode'] == 'teacher':
        logger.info("--- Running in TEACHER MODE ---")
        train_teacher_mode(config, device, models_dir, plot_dir, df_hist, df_offline, df_online, df_test, num_cols, cat_cols, vocab, vocab_sizes, logger)
    elif config['mode'] == 'student':
        logger.info("--- Running in STUDENT MODE ---")
        train_student_mode(config, device, models_dir, plot_dir, df_offline, df_online, df_test, num_cols, cat_cols, vocab, vocab_sizes, logger)
    elif config['mode'] == 'tskd':
        logger.info("--- Running in TSKD FRAMEWORK MODE ---")
        train_tskd_mode(config, device, models_dir, plot_dir, df_offline, df_online, df_test, num_cols, cat_cols, vocab, vocab_sizes, logger)
    return


def train_teacher_mode(config, device, models_dir, plot_dir, df_hist, df_offline, df_online, df_test, num_cols, cat_cols, vocab, vocab_sizes, logger):
    """Train teacher models in teacher mode."""
    os.makedirs(models_dir, exist_ok=True)
    aucs1, losses1 = [], []
    aucs2, losses2 = [], []
    aucs3, losses3 = [], []
    times1, times2, times3 = [], [], []
    test_loader = create_dataloader(config, df_test, 'test_loader', num_cols, cat_cols, vocab, logger)
    for i in range(config['num_runs']):
        set_seed(i)
        logger.info(f"Training Historical Teacher #{i+1}...")
        teacher_info = generate_model_info_string(config, 'teacher')
        model_path = os.path.join(models_dir, f"teacher_model_{teacher_info}_{i}.pth")
        teacher_model = CTRModel(config['teacher_arch'], vocab_sizes, num_cols, config['teacher_embedding_dim'], config['hidden_dims_teacher']).to(device)
        
        # 创建分离的优化器
        embedding_optimizer, dense_optimizer = create_separate_optimizers(
            teacher_model, config['lr_sparse'], config['lr_dense'], logger
        )

        loss_history = []
        samples_processed = 0
        cumulative_time = 0.0  # 累计训练时间（秒），用于 times1/2/3 的累加

        # 进度记录开关与容器（跨 hist/offline/online 合并记录）
        progress_enabled = bool(config.get('log_progress_100', False))
        progress_lists = {'time_list': [], 'auc_list': [], 'logloss_list': []}
        global_step = 0
        run_start = time.perf_counter()
        eval_overhead = 0.0

        if config['hist']:
            teacher_model.train()
            hist_loader = create_dataloader(config, df_hist, 'hist_loader', num_cols, cat_cols, vocab, logger)
            t0 = time.perf_counter()
            for x_num, x_cat, y_true in tqdm(hist_loader, desc=f"Training Teacher {i+1}", leave=True):
                x_num, x_cat, y_true = x_num.to(device, non_blocking=True), x_cat.to(device, non_blocking=True), y_true.to(device, non_blocking=True)
                embedding_optimizer.zero_grad()
                dense_optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(teacher_model(x_num, x_cat), y_true)
                loss.backward()
                embedding_optimizer.step()
                dense_optimizer.step()
                samples_processed += len(y_true)
                loss_history.append((samples_processed, loss.item()))
                global_step += 1
                if progress_enabled and (global_step % 100 == 0):
                    t_eval0 = time.perf_counter()
                    auc_tmp, loss_tmp = evaluate_model(teacher_model, test_loader, device)
                    t_eval = time.perf_counter() - t_eval0
                    eval_overhead += t_eval
                    elapsed_no_eval = time.perf_counter() - run_start - eval_overhead
                    progress_lists['time_list'].append(float(elapsed_no_eval))
                    progress_lists['auc_list'].append(float(auc_tmp))
                    progress_lists['logloss_list'].append(float(loss_tmp))
            train_time_hist = time.perf_counter() - t0
            cumulative_time += train_time_hist

            auc, loss = evaluate_model(teacher_model, test_loader, device)
            logger.info(f"Teacher #{i+1} - AUC: {auc:.2f}, LogLoss: {loss:.4f}, TrainTime(s): {train_time_hist:.1f}")
            aucs1.append(auc)
            losses1.append(loss)
            times1.append(cumulative_time)
            # 阶段结束也记录一次
            if progress_enabled:
                t_eval0 = time.perf_counter()
                auc_tmp, loss_tmp = evaluate_model(teacher_model, test_loader, device)
                t_eval = time.perf_counter() - t_eval0
                eval_overhead += t_eval
                elapsed_no_eval = time.perf_counter() - run_start - eval_overhead
                progress_lists['time_list'].append(float(elapsed_no_eval))
                progress_lists['auc_list'].append(float(auc_tmp))
                progress_lists['logloss_list'].append(float(loss_tmp))
            del hist_loader  # 释放内存
            force_garbage_collection()

        if config['offline']:
            teacher_model.train()
            offline_loader = create_dataloader(config, df_offline, 'offline_loader', num_cols, cat_cols, vocab, logger)
            t0 = time.perf_counter()
            for x_num, x_cat, y_true in tqdm(offline_loader, desc=f"Training Teacher {i+1}", leave=True):
                x_num, x_cat, y_true = x_num.to(device, non_blocking=True), x_cat.to(device, non_blocking=True), y_true.to(device, non_blocking=True)
                embedding_optimizer.zero_grad()
                dense_optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(teacher_model(x_num, x_cat), y_true)
                loss.backward()
                embedding_optimizer.step()
                dense_optimizer.step()
                samples_processed += len(y_true)
                loss_history.append((samples_processed, loss.item()))
                global_step += 1
                if progress_enabled and (global_step % 100 == 0):
                    t_eval0 = time.perf_counter()
                    auc_tmp, loss_tmp = evaluate_model(teacher_model, test_loader, device)
                    t_eval = time.perf_counter() - t_eval0
                    eval_overhead += t_eval
                    elapsed_no_eval = time.perf_counter() - run_start - eval_overhead
                    progress_lists['time_list'].append(float(elapsed_no_eval))
                    progress_lists['auc_list'].append(float(auc_tmp))
                    progress_lists['logloss_list'].append(float(loss_tmp))
            train_time_offline = time.perf_counter() - t0
            cumulative_time += train_time_offline
            
            auc, loss = evaluate_model(teacher_model, test_loader, device)
            logger.info(f"Teacher #{i+1} - AUC: {auc:.2f}, LogLoss: {loss:.4f}, TrainTime(s): {train_time_offline:.1f}")
            aucs2.append(auc)
            losses2.append(loss)
            times2.append(cumulative_time)
            if progress_enabled:
                t_eval0 = time.perf_counter()
                auc_tmp, loss_tmp = evaluate_model(teacher_model, test_loader, device)
                t_eval = time.perf_counter() - t_eval0
                eval_overhead += t_eval
                elapsed_no_eval = time.perf_counter() - run_start - eval_overhead
                progress_lists['time_list'].append(float(elapsed_no_eval))
                progress_lists['auc_list'].append(float(auc_tmp))
                progress_lists['logloss_list'].append(float(loss_tmp))
            del offline_loader  # 释放内存
            force_garbage_collection()

            # 根据配置决定是否保存教师模型和优化器状态
            if config['update_teacher_models']:
                # 保存模型和优化器状态
                torch.save({
                    'model_state_dict': teacher_model.state_dict(),
                    'embedding_optimizer_state_dict': embedding_optimizer.state_dict(),
                    'dense_optimizer_state_dict': dense_optimizer.state_dict()
                }, model_path)
                logger.info(f"Teacher model and optimizer states saved to: {model_path}")
            else:
                logger.info(f"Skipping teacher model save (update_teacher_models=False)")

        if config['online']:
            teacher_model.train()
            online_loader = create_dataloader(config, df_online, 'online_loader', num_cols, cat_cols, vocab, logger)
            t0 = time.perf_counter()
            for x_num, x_cat, y_true in tqdm(online_loader, desc=f"Training Teacher {i+1}", leave=True):
                x_num, x_cat, y_true = x_num.to(device, non_blocking=True), x_cat.to(device, non_blocking=True), y_true.to(device, non_blocking=True)
                embedding_optimizer.zero_grad()
                dense_optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(teacher_model(x_num, x_cat), y_true)
                loss.backward()
                embedding_optimizer.step()
                dense_optimizer.step()
                samples_processed += len(y_true)
                loss_history.append((samples_processed, loss.item()))
                global_step += 1
                if progress_enabled and (global_step % 100 == 0):
                    t_eval0 = time.perf_counter()
                    auc_tmp, loss_tmp = evaluate_model(teacher_model, test_loader, device)
                    t_eval = time.perf_counter() - t_eval0
                    eval_overhead += t_eval
                    elapsed_no_eval = time.perf_counter() - run_start - eval_overhead
                    progress_lists['time_list'].append(float(elapsed_no_eval))
                    progress_lists['auc_list'].append(float(auc_tmp))
                    progress_lists['logloss_list'].append(float(loss_tmp))
            train_time_online = time.perf_counter() - t0
            cumulative_time += train_time_online
            auc, loss = evaluate_model(teacher_model, test_loader, device)
            logger.info(f"Teacher #{i+1} - AUC: {auc:.2f}, LogLoss: {loss:.4f}, TrainTime(s): {train_time_online:.1f}")
            aucs3.append(auc)
            losses3.append(loss)
            times3.append(cumulative_time)
            if progress_enabled:
                t_eval0 = time.perf_counter()
                auc_tmp, loss_tmp = evaluate_model(teacher_model, test_loader, device)
                t_eval = time.perf_counter() - t_eval0
                eval_overhead += t_eval
                elapsed_no_eval = time.perf_counter() - run_start - eval_overhead
                progress_lists['time_list'].append(float(elapsed_no_eval))
                progress_lists['auc_list'].append(float(auc_tmp))
                progress_lists['logloss_list'].append(float(loss_tmp))
        # per-run artifacts and cleanup
        plot_loss_curve(loss_history, f"Historical_Teacher_{teacher_info}_{i+1}", plot_dir)

        del teacher_model, embedding_optimizer, dense_optimizer, loss_history, online_loader
        force_garbage_collection()
        _log_memory(logger, f"after TEACHER run {i+1}")

        # 将本 run 进度列表写入结果文件
        if progress_enabled:
            try:
                progress_line = (
                    f"RUN {i+1} PROGRESS:"
                    f"\n time_list={progress_lists['time_list']}, "
                    f"\n auc_list={progress_lists['auc_list']}, "
                    f"\n logloss_list={progress_lists['logloss_list']}"
                )
                logger.info(progress_line)
                _append_to_result_file(progress_line)
            except Exception:
                pass

    teacher_summary_line = (
        f"Average AUC, LogLoss, Time(s): "
        f"{np.mean(aucs1):.2f}±{np.std(aucs1):.2f},{np.mean(losses1):.4f}±{np.std(losses1):.4f},{np.mean(times1):.1f}±{np.std(times1):.1f},"
        f"{np.mean(aucs2):.2f}±{np.std(aucs2):.2f},{np.mean(losses2):.4f}±{np.std(losses2):.4f},{np.mean(times2):.1f}±{np.std(times2):.1f},"
        f"{np.mean(aucs3):.2f}±{np.std(aucs3):.2f},{np.mean(losses3):.4f}±{np.std(losses3):.4f},{np.mean(times3):.1f}±{np.std(times3):.1f}"
    )
    logger.info(teacher_summary_line)
    _append_to_result_file(teacher_summary_line)


def train_student_mode(config, device, models_dir, plot_dir, df_offline, df_online, df_test, num_cols, cat_cols, vocab, vocab_sizes, logger):
    """Train student models in student mode."""
    os.makedirs(models_dir, exist_ok=True)
    aucs1, losses1 = [], []
    aucs2, losses2 = [], []
    times1, times2 = [], []
    test_loader = create_dataloader(config, df_test, 'test_loader', num_cols, cat_cols, vocab, logger)
    for i in range(config['num_runs']):
        set_seed(i)
        logger.info(f"Training Student #{i+1}...")
        student_info = generate_model_info_string(config, 'student')
        student_model = CTRModel(config['student_arch'], vocab_sizes, num_cols, config['student_embedding_dim'], config['hidden_dims_student']).to(device)
        
        # 创建分离的优化器
        embedding_optimizer, dense_optimizer = create_separate_optimizers(
            student_model, config['lr_sparse'], config['lr_dense'], logger
        )

        loss_history = []
        samples_processed = 0
        cumulative_time = 0.0

        # 进度记录开关与容器（跨 offline/online 合并记录）
        progress_enabled = bool(config.get('log_progress_100', False))
        progress_lists = {'time_list': [], 'auc_list': [], 'logloss_list': []}
        global_step = 0
        run_start = time.perf_counter()
        eval_overhead = 0.0
        if config['offline']:
            student_model.train()
            offline_loader = create_dataloader(config, df_offline, 'offline_loader', num_cols, cat_cols, vocab, logger)
            t0 = time.perf_counter()
            for x_num, x_cat, y_true in tqdm(offline_loader, desc=f"Training Student {i+1}", leave=True):
                x_num, x_cat, y_true = x_num.to(device, non_blocking=True), x_cat.to(device, non_blocking=True), y_true.to(device, non_blocking=True)
                embedding_optimizer.zero_grad()
                dense_optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(student_model(x_num, x_cat), y_true)
                loss.backward()
                embedding_optimizer.step()
                dense_optimizer.step()
                samples_processed += len(y_true)
                loss_history.append((samples_processed, loss.item()))
                global_step += 1
                if progress_enabled and (global_step % 100 == 0):
                    t_eval0 = time.perf_counter()
                    auc_tmp, loss_tmp = evaluate_model(student_model, test_loader, device)
                    t_eval = time.perf_counter() - t_eval0
                    eval_overhead += t_eval
                    elapsed_no_eval = time.perf_counter() - run_start - eval_overhead
                    progress_lists['time_list'].append(float(elapsed_no_eval))
                    progress_lists['auc_list'].append(float(auc_tmp))
                    progress_lists['logloss_list'].append(float(loss_tmp))
            train_time_offline = time.perf_counter() - t0

            auc, loss = evaluate_model(student_model, test_loader, device)
            logger.info(f"Student #{i+1} - AUC: {auc:.2f}, LogLoss: {loss:.4f}, TrainTime(s): {train_time_offline:.1f}")
            aucs1.append(auc)
            losses1.append(loss)
            cumulative_time += train_time_offline
            times1.append(cumulative_time)
            if progress_enabled:
                t_eval0 = time.perf_counter()
                auc_tmp, loss_tmp = evaluate_model(student_model, test_loader, device)
                t_eval = time.perf_counter() - t_eval0
                eval_overhead += t_eval
                elapsed_no_eval = time.perf_counter() - run_start - eval_overhead
                progress_lists['time_list'].append(float(elapsed_no_eval))
                progress_lists['auc_list'].append(float(auc_tmp))
                progress_lists['logloss_list'].append(float(loss_tmp))
            del offline_loader  # 释放内存
            force_garbage_collection()

        if config['online']:
            student_model.train()
            online_loader = create_dataloader(config, df_online, 'online_loader', num_cols, cat_cols, vocab, logger)
            t0 = time.perf_counter()
            for x_num, x_cat, y_true in tqdm(online_loader, desc=f"Training Student {i+1}", leave=True):
                x_num, x_cat, y_true = x_num.to(device, non_blocking=True), x_cat.to(device, non_blocking=True), y_true.to(device, non_blocking=True)
                embedding_optimizer.zero_grad()
                dense_optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(student_model(x_num, x_cat), y_true)
                loss.backward()
                embedding_optimizer.step()
                dense_optimizer.step()
                samples_processed += len(y_true)
                loss_history.append((samples_processed, loss.item()))
                global_step += 1
                if progress_enabled and (global_step % 100 == 0):
                    t_eval0 = time.perf_counter()
                    auc_tmp, loss_tmp = evaluate_model(student_model, test_loader, device)
                    t_eval = time.perf_counter() - t_eval0
                    eval_overhead += t_eval
                    elapsed_no_eval = time.perf_counter() - run_start - eval_overhead
                    progress_lists['time_list'].append(float(elapsed_no_eval))
                    progress_lists['auc_list'].append(float(auc_tmp))
                    progress_lists['logloss_list'].append(float(loss_tmp))
            train_time_online = time.perf_counter() - t0
            auc, loss = evaluate_model(student_model, test_loader, device)
            logger.info(f"Student #{i+1} - AUC: {auc:.2f}, LogLoss: {loss:.4f}, TrainTime(s): {train_time_online:.1f}")
            aucs2.append(auc)
            losses2.append(loss)
            cumulative_time += train_time_online
            times2.append(cumulative_time)
            if progress_enabled:
                t_eval0 = time.perf_counter()
                auc_tmp, loss_tmp = evaluate_model(student_model, test_loader, device)
                t_eval = time.perf_counter() - t_eval0
                eval_overhead += t_eval
                elapsed_no_eval = time.perf_counter() - run_start - eval_overhead
                progress_lists['time_list'].append(float(elapsed_no_eval))
                progress_lists['auc_list'].append(float(auc_tmp))
                progress_lists['logloss_list'].append(float(loss_tmp))
        # per-run artifacts and cleanup
        plot_loss_curve(loss_history, f"Student_{student_info}_{i+1}", plot_dir)

        del student_model, embedding_optimizer, dense_optimizer, loss_history, online_loader
        force_garbage_collection()
        _log_memory(logger, f"after STUDENT run {i+1}")

        # 将本 run 进度列表写入结果文件
        if progress_enabled:
            try:
                progress_line = (
                    f"RUN {i+1} PROGRESS:"
                    f"\n time_list={progress_lists['time_list']}, "
                    f"\n auc_list={progress_lists['auc_list']}, "
                    f"\n logloss_list={progress_lists['logloss_list']}"
                )
                logger.info(progress_line)
                _append_to_result_file(progress_line)
            except Exception:
                pass
    

    student_summary_line = (
        f"Average AUC, LogLoss, Time(s): "
        f"{np.mean(aucs1):.2f}±{np.std(aucs1):.2f},{np.mean(losses1):.4f}±{np.std(losses1):.4f},{np.mean(times1):.1f}±{np.std(times1):.1f},"
        f"{np.mean(aucs2):.2f}±{np.std(aucs2):.2f},{np.mean(losses2):.4f}±{np.std(losses2):.4f},{np.mean(times2):.1f}±{np.std(times2):.1f}"
    )
    logger.info(student_summary_line)
    _append_to_result_file(student_summary_line)


def train_tskd_mode(config, device, models_dir, plot_dir, df_offline, df_online, df_test, num_cols, cat_cols, vocab, vocab_sizes, logger):
    # --- Train or Load Historical Teacher Models ---
    logger.info("--- Preparing Historical Teacher Models ---")
    os.makedirs(models_dir, exist_ok=True)

    all_run_results = []
    test_loader = create_dataloader(config, df_test, 'test_loader', num_cols, cat_cols, vocab, logger)
    # --- Start Multiple Experiment Runs ---
    for run_idx in range(config['num_runs']):
        set_seed(run_idx)
        logger.info(f"{'='*30} STARTING RUN {run_idx + 1}/{config['num_runs']} {'='*30}")
        
        teacher_info = generate_model_info_string(config, 'teacher')
        model_path = os.path.join(models_dir, f"teacher_model_{teacher_info}_{run_idx}.pth")
        # Build teacher model via factory
        teacher_model = CTRModel(config['teacher_arch'], vocab_sizes, num_cols, config['teacher_embedding_dim'], config['hidden_dims_teacher']).to(device)

        # 检查教师模型文件是否存在
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            teacher_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded existing teacher model from: {model_path}")
            
            # 检查是否包含优化器状态
            if 'embedding_optimizer_state_dict' in checkpoint and 'dense_optimizer_state_dict' in checkpoint:
                logger.info("Found optimizer states in checkpoint, will use continuous training strategy")
                teacher_optimizer_states = {
                    'embedding_optimizer_state_dict': checkpoint['embedding_optimizer_state_dict'],
                    'dense_optimizer_state_dict': checkpoint['dense_optimizer_state_dict']
                }
            else:
                logger.info("No optimizer states found in checkpoint, will use fresh optimizers")
                teacher_optimizer_states = None
        else:
            logger.warning(f"Teacher model file not found: {model_path}")
            raise FileNotFoundError(f"Teacher model file not found: {model_path}. Please run teacher mode first or set update_teacher_models=True.")
        
        teacher_model = teacher_model.to(device)
        # auc, loss = evaluate_model(teacher_model, test_loader, device)
        # logger.info(f"Teacher - AUC, LogLoss: {auc:.2f}, {loss:.4f}")
        # exit()

        run_models_dir = os.path.join(models_dir, f"run_{run_idx+1}")
        os.makedirs(run_models_dir, exist_ok=True)

        # Initialize models for this run via factory
        student_model = CTRModel(config['student_arch'], vocab_sizes, num_cols, config['student_embedding_dim'], config['hidden_dims_student']).to(device)
        tskd_pipeline = TSKD_Framework(student_model, teacher_model, config, device, teacher_optimizer_states, num_cols, cat_cols, vocab)

        # Only keep S2 student metrics; S1 student and S2 teacher metrics are not computed
        s2_auc, s2_loss = 0, 0
        # Initialize training time containers per run (exclude evaluation)
        s1_train_time = 0.0
        s2_train_time = 0.0

        # --- Stage 1 ---
        logger.info("--- Running Stage 1 ---")
        student_info = generate_model_info_string(config, 'student')
        s1_model_path = os.path.join(run_models_dir, f"student_model_s1_{student_info}.pth")
        if os.path.exists(s1_model_path) and not config['force_retrain_s1']:
            logger.info(f"--- Loading Stage 1 student model for run {run_idx+1} from {s1_model_path} ---")
            student_model.load_state_dict(torch.load(s1_model_path, map_location=device))
            s1_train_time = 0.0  # no training performed in S1 for this run
        else:
            logger.info("--- Preparing Representative Sample Set (D_rep) for Stage 1 ---")
            # df_offline = pd.concat([df_offline, df_online], ignore_index=True)
            offline_loader = create_dataloader(config, df_offline, 'offline_loader', num_cols, cat_cols, vocab, logger)
            # 进度评估：每100个batch记录一次(可选)，S1 与 S2 共用同一容器
            progress_enabled = bool(config.get('log_progress_100', False))
            progress_cfg = {'enabled': progress_enabled, 'interval': 100}
            progress_lists = {'time_list': [], 'auc_list': [], 'logloss_list': []}
            s1_train_time = tskd_pipeline.stage1(
                df_offline, offline_loader, logger, plot_dir, run_idx + 1, student_info,
                eval_loader=test_loader, progress_log_config=progress_cfg, time_auc_log=progress_lists
            )
            torch.save(student_model.state_dict(), s1_model_path)
            del offline_loader  # 释放内存
            force_garbage_collection()

        # Evaluate student model after Stage 1 and log with training time
        s1_auc, s1_loss = evaluate_model(student_model, test_loader, device)
        logger.info(f"Run {run_idx+1} | Student (after S1) - AUC: {s1_auc:.2f}, LogLoss: {s1_loss:.4f}, TrainTime(s): {s1_train_time:.1f}")

        # --- Stage 2 ---
        if not config['skip_s2']:
            logger.info("--- Running Stage 2: Live Co-evolutionary Alignment ---")
            
            # 创建ReplayDataset（如果启用replay功能）
            if config['replay']:
                from TSKD.dataset import ReplayDataset
                replay_ratio = float(config.get('replay_ratio', 0.1))
                replay_batch_size = max(1, int(config['batch_size'] * replay_ratio))  # 可配置的比例，至少为1
                if isinstance(df_offline, SplitHandle):
                    replay_dataset = StreamingReplayDataset(df_offline, num_cols, cat_cols, config['dataset'], replay_batch_size)
                    logger.info(f"创建StreamingReplayDataset，每次采样{replay_batch_size}个样本")
                else:
                    replay_dataset = ReplayDataset(df_offline, num_cols, cat_cols, vocab, config['dataset'], replay_batch_size)
                    logger.info(f"创建ReplayDataset，每次采样{replay_batch_size}个样本")
            else:
                replay_dataset = None

            online_loader = create_dataloader(config, df_online, 'online_loader', num_cols, cat_cols, vocab, logger)
            # 注意：进度容器 progress_lists 复用 Stage1 的，以便合并输出；若S1跳过训练（加载模型），则此处需初始化
            if 'progress_lists' not in locals():
                progress_lists = {'time_list': [], 'auc_list': [], 'logloss_list': []}
                progress_cfg = {'enabled': bool(config.get('log_progress_100', False)), 'interval': 100}
            s2_train_time = tskd_pipeline.stage2(
                online_loader,
                replay_dataset,
                logger,
                plot_dir,
                run_idx + 1,
                student_info,
                teacher_info,
                eval_loader=test_loader,
                progress_log_config=progress_cfg,
                time_auc_log=progress_lists,
                time_offset=s1_train_time,
            )
            del online_loader  # 释放内存
            force_garbage_collection()

            s2_auc, s2_loss = evaluate_model(student_model, test_loader, device)
            logger.info(f"Run {run_idx+1} | Student (after S2) - AUC: {s2_auc:.2f}, LogLoss: {s2_loss:.4f}, TrainTime(s): {s2_train_time:.1f}")

        # Store results
        all_run_results.append({
            'run': run_idx + 1,
            's1_student_auc': s1_auc, 's1_student_loss': s1_loss,
            's1_time': s1_train_time,
            's2_student_auc': s2_auc, 's2_student_loss': s2_loss,
            'total_time': (s1_train_time + s2_train_time),
        })
        # 如果启用了进度记录，把本次 run 的列表也收集起来，便于统一写入 result.txt（S1+S2 合并）
        if config.get('log_progress_100', False):
            # 将列表持久化附加到结果文件中（逐run输出，避免占用过多内存）
            try:
                progress_line = (
                    f"RUN {run_idx+1} PROGRESS:"
                    f"\n time_list={progress_lists['time_list']}"
                    f"\n auc_list={progress_lists['auc_list']}"
                    f"\n logloss_list={progress_lists['logloss_list']}"
                )
                logger.info(progress_line)
                _append_to_result_file(progress_line)
            except Exception:
                pass
    
        # 清理本次 run 的大对象，释放显存/内存，避免在下一次 run 时累积
        del student_model, teacher_model, tskd_pipeline, replay_dataset
        force_garbage_collection()
        _log_memory(logger, f"after TSKD run {run_idx+1}")

    # --- Final Aggregated Report ---
    df_results = pd.DataFrame(all_run_results)
    logger.info("="*70)
    logger.info("--- DETAILED PERFORMANCE ACROSS ALL RUNS ---")
    logger.info(f"{df_results.round(4).to_string()}")
    
    logger.info("="*70)
    logger.info("--- AGGREGATED PERFORMANCE METRICS (Mean ± Std Dev) ---")
    summary = {
        'Metric': ['AUC', 'LogLoss', 'TrainTime(s)'],
        'Student (after S1)': [
            f"{df_results['s1_student_auc'].mean():.2f}±{df_results['s1_student_auc'].std():.2f}",
            f"{df_results['s1_student_loss'].mean():.4f}±{df_results['s1_student_loss'].std():.4f}",
            f"{df_results['s1_time'].mean():.1f}±{df_results['s1_time'].std():.1f}"
        ],
        'Student (after S2)': [
            f"{df_results['s2_student_auc'].mean():.2f}±{df_results['s2_student_auc'].std():.2f}",
            f"{df_results['s2_student_loss'].mean():.4f}±{df_results['s2_student_loss'].std():.4f}",
            f"{df_results['total_time'].mean():.1f}±{df_results['total_time'].std():.1f}"
        ]
    }
    df_summary = pd.DataFrame(summary).set_index('Metric').transpose()
    logger.info(f"{df_summary}")
    # Flat summary line: S1[AUC,LogLoss,Time], S2-Student[AUC,LogLoss,TotalTime]
    flat_summary_line = (
        f"{df_summary.iloc[0, 0]}, {df_summary.iloc[0, 1]}, {df_summary.iloc[0, 2]}, "
        f"{df_summary.iloc[1, 0]}, {df_summary.iloc[1, 1]}, {df_summary.iloc[1, 2]}"
    )
    logger.info(flat_summary_line)
    _append_to_result_file(flat_summary_line)
    logger.info("="*70)
    logger.info("--- TS-KD Framework execution completed ---")