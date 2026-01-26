# coding:utf-8 
import torch
import os
import argparse
import logging
from datetime import datetime
from TSKD import train, DataProcessor
import resource
import signal
import os
import numpy as np
import yaml


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TS-KD Framework for CTR Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 基础配置参数
    parser.add_argument('--mode', type=str, choices=['teacher', 'student', 'tskd'], 
                        help="运行模式: 'teacher'=仅训练教师模型, 'student'=仅训练学生模型, 'tskd'=完整TSKD框架")
    parser.add_argument('--dataset', type=str, choices=['avazu', 'criteo', 'criteo_ctr_full'], help="数据集选择: 'avazu' 或 'criteo'")
    parser.add_argument('--num_runs', type=int, help="实验运行次数，用于结果平均")
    parser.add_argument('--stage2_update_mode', type=str, choices=['both', 'student'], help="训练模式: 'both'=同时训练学生和教师模型, 'student'=仅训练学生模型")
    parser.add_argument('--log_progress_100', type=str_to_bool, help="是否每100个batch记录一次(不含评估时间)并进行评估")

    # 教师模型训练参数
    parser.add_argument('--hist', type=str_to_bool, help="使用历史数据训练教师模型")
    parser.add_argument('--offline', type=str_to_bool, help="使用离线数据训练")
    parser.add_argument('--online', type=str_to_bool, help="使用在线数据训练")
    parser.add_argument('--update_teacher_models', type=str_to_bool, help="更新并保存历史教师模型")

    # 模型架构参数
    parser.add_argument('--teacher_arch', type=str, 
                        choices=['deepfm', 'dcn', 'mlp', 'widedeep', 'xdeepfm', 'autoint', 'fibi', 'afm', 'nfm'], help="教师模型架构")
    parser.add_argument('--student_arch', type=str, 
                        choices=['deepfm', 'dcn', 'mlp', 'widedeep', 'xdeepfm', 'autoint', 'fibi', 'afm', 'nfm'], help="学生模型架构")
    parser.add_argument('--teacher_embedding_dim', type=int, help="教师模型嵌入维度")
    parser.add_argument('--student_embedding_dim', type=int, help="学生模型嵌入维度")
    parser.add_argument('--hidden_dims_teacher', nargs='+', type=int, help="教师模型隐藏层维度")
    parser.add_argument('--hidden_dims_student', nargs='+', type=int, help="学生模型隐藏层维度")

    # 学习率参数
    parser.add_argument('--lr_sparse', type=float, help="稀疏嵌入参数学习率")
    parser.add_argument('--lr_dense', type=float, help="稠密模型参数学习率")

    # Stage 1 参数
    parser.add_argument('--force_retrain_s1', type=str_to_bool, help="重新训练Stage 1学生模型")
    parser.add_argument('--s1_dense_epochs', type=int, help="Stage 1B: 冻结嵌入后的模型蒸馏轮数")
    parser.add_argument('--s1_joint_epochs', type=int, help="Stage 1C: 嵌入和稠密模型联合训练轮数")
    parser.add_argument('--s1a_method', type=str, choices=['orthomap', 'relation', 'hint'], help="Stage 1A方法选择")
    parser.add_argument('--alpha_1b', type=float, help="Stage 1B 蒸馏损失权重（KD vs BCE）")
    parser.add_argument('--alpha_1c', type=float, help="Stage 1C 蒸馏损失权重（KD vs BCE）")

    # 样本选择参数
    parser.add_argument('--sample_ratio', type=float, help="Stage 1代表性样本比例")
    parser.add_argument('--sample_selection', type=str, 
                        choices=['random', 'biased_sample', 'temporal', 'recent', 'all', 'all_shuffle', 'weighted', 'combined', 'generate'], 
                        help="Stage 1代表性样本选择方法")
    parser.add_argument('--target_pos_rate', type=float, help="biased_sample方法的目标正样本率")
    parser.add_argument('--s1c_sparse_l2', type=float, help="Stage 1C中稀疏嵌入L2正则化权重")

    # Stage 2 参数
    parser.add_argument('--replay', type=str_to_bool, help="在Stage 2中启用重放机制")
    parser.add_argument('--alpha_2', type=float, help="Stage 2中稀疏嵌入蒸馏损失的权重")
    parser.add_argument('--teacher_update_freq', type=int, help="教师模型更新频率 (轮数)")
    parser.add_argument('--replay_ratio', type=float, help="Replay 批比例（相对 batch_size 的比例）")
    
    # 蒸馏相关超参数（Hint-KD / RKD）
    parser.add_argument('--temperature', type=float, help="蒸馏温度系数（soft targets 平滑温度）")
    parser.add_argument('--distill_loss', type=str, choices=['kl', 'huber'], help="蒸馏损失类型: 'kl' (默认) 或 'huber'（在soft概率上使用Huber/SmoothL1）")
    parser.add_argument('--use_hint_kd', type=str_to_bool, help="是否启用 Hint-KD（在稀疏 embedding 上使用 projection + MSE 对齐）")
    parser.add_argument('--hint_lambda', type=float, help="Hint-KD 损失权重")
    parser.add_argument('--use_rkd', type=str_to_bool, help="是否启用 RKD（关系知识蒸馏：distance + angle）")
    parser.add_argument('--rkd_lambda', type=float, help="RKD 损失的整体缩放系数")

    # 阶段跳过参数
    parser.add_argument('--skip_s1a', type=str_to_bool, help="跳过Stage 1A (稀疏嵌入InfoNCE)")
    parser.add_argument('--skip_s1b', type=str_to_bool, help="跳过Stage 1B (稠密模型知识蒸馏)")
    parser.add_argument('--skip_s1c', type=str_to_bool, help="跳过Stage 1C (嵌入和稠密模型联合训练)")
    parser.add_argument('--skip_s2', type=str_to_bool, help="跳过Stage 2 (模型蒸馏)")

    # 模型特定参数
    parser.add_argument('--num_cross_layers', type=int, help="DCN模型的交叉层数量")
    parser.add_argument('--num_cin_layers', type=int, help="xDeepFM模型的CIN层数量")
    parser.add_argument('--num_attention_layers', type=int, help="AutoInt模型的注意力层数量")
    parser.add_argument('--num_attention_heads', type=int, help="AutoInt模型的注意力头数量")

    # 资源限制参数
    parser.add_argument('--max_memory_gb', type=int, help="最大内存使用限制 (GB)")
    parser.add_argument('--max_cpu_time_hour', type=int, help="最大CPU时间限制 (小时)")
    
    args = parser.parse_args()
    return args


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_yaml_config(config_path):
    """加载YAML配置文件并转换为扁平字典格式"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 将嵌套的配置转换为扁平字典
    flat_config = {}
    for key in (
            'basic', 'data', 'teacher_training', 'model_architecture', 'learning_rate',
            'stage1', 'sample_selection', 'stage2', 'distillation', 'skip_stages', 'model_specific', 'resource_limits'
        ):
            if key in config:
                flat_config.update(config[key])
    
    return flat_config


def set_limits(max_memory_gb=28, max_cpu_time_hour=3):
    """在 Python 内部设置资源限制"""
    
    # --- 限制最大内存使用 (虚拟内存) ---
    # # 将 MB 转换为 Bytes
    # max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
    
    # # setrlimit 需要两个参数：(soft limit, hard limit)
    # # soft limit 是内核强制执行的限制
    # # hard limit 是 soft limit 可以被提高到的最大值
    # try:
    #     resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
    #     print(f"成功设置内存限制为: {max_memory_gb} GB")
    # except Exception as e:
    #     print(f"设置内存限制失败: {e} (可能权限不足)")

    # --- 限制最大 CPU 时间 ---
    def cpu_time_exceeded_handler(signum, frame):
        """当 CPU 时间超限时，此函数被调用"""
        print("CPU 时间限制已达到，进程即将终止。")
        # 在这里可以执行一些清理操作
        raise SystemExit("CPU time limit exceeded")

    # 注册信号处理器，当 CPU 时间超过 soft limit 时，内核会发送 SIGXCPU 信号
    signal.signal(signal.SIGXCPU, cpu_time_exceeded_handler)
    
    try:
        max_cpu_time_sec = max_cpu_time_hour * 3600
        resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_time_sec, max_cpu_time_sec))
        print(f"成功设置 CPU 时间限制为: {max_cpu_time_sec} 秒")
    except Exception as e:
        print(f"设置 CPU 时间限制失败: {e}")


def setup_logging(config, mode):
    """Setup logging configuration."""
    # Create logs directory
    logs_dir = f"{config['dataset']}_logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log filename with timestamp and mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f"{config['dataset']}_{mode}_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return logging.getLogger(__name__)


if __name__ == '__main__':
    args = parse_args()
    
    # 加载YAML配置
    if os.path.exists('config.yaml'):
        yaml_config = load_yaml_config('config.yaml')
        print(f"已加载YAML配置文件: config.yaml")
    else:
        print(f"警告: 配置文件 config.yaml 不存在，使用默认参数")
        yaml_config = {}
    
    # 将命令行参数转换为字典
    cmd_args = vars(args)
    
    # 合并配置：YAML配置作为基础，命令行参数覆盖YAML配置
    config = yaml_config.copy()
    
    # 记录被命令行参数覆盖的YAML配置
    overridden_params = []
    
    # 命令行参数覆盖YAML配置（仅当命令行参数不为None时）
    for key, value in cmd_args.items():
        if value is not None:  # 如果命令行参数有值
            if key in yaml_config and yaml_config[key] != value:
                overridden_params.append(f"{key}: {yaml_config[key]} -> {value}")
            config[key] = value
    
    # 输出配置覆盖信息
    if overridden_params:
        print("命令行参数覆盖的YAML配置:")
        for param in overridden_params:
            print(f"  - {param}")
    else:
        print("没有YAML配置被命令行参数覆盖")
    
    # 设置资源限制
    set_limits(
        max_memory_gb=config.get('max_memory_gb', 28),
        max_cpu_time_hour=config.get('max_cpu_time_hour', 3)
    )
    
    # Setup logging
    logger = setup_logging(config, config['mode'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Log GPU information if available
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("CUDA not available, using CPU. Training will be slower.")

    # 允许通过配置开启全量内存模式（Parquet -> DataFrame）
    full_in_memory = bool(config.get('full_in_memory', False))
    df_hist, df_offline, df_online, df_test, num_cols, cat_cols, vocab, vocab_sizes = DataProcessor(config['dataset'], full_in_memory=full_in_memory).process_data()
    logger.info(f"Vocabulary sizes: {vocab_sizes}")
    logger.info(f"Data shapes - Hist: {len(df_hist)}, Offline: {len(df_offline)}, Online: {len(df_online)}, Test: {len(df_test)}")
    
    train(config, device, df_hist, df_offline, df_online, df_test, num_cols, cat_cols, vocab, vocab_sizes, logger)
