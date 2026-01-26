import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import os
import time
from .dataset import MyDataset, DynamicSamplingDataset
from .dataset_stream import SplitHandle, StreamingHDF5IndexDataset, StreamingIndexManager, StreamingSamplerCollate, _DummyIndexDataset


def create_separate_optimizers(model, lr_sparse, lr_dense, logger=None):
    """
    为模型创建分离的优化器：
    - sparse embedding参数使用Adam优化器
    - 稠密模型参数使用Adam优化器
    
    Args:
        model: 模型对象
        lr_sparse: sparse embedding参数的学习率 (默认0.01)
        lr_dense: 稠密模型参数的学习率 (默认0.001)
        logger: 日志记录器
    
    Returns:
        tuple: (embedding_optimizer, dense_optimizer)
    """
    # 获取sparse embedding参数
    embedding_params = []
    dense_params = []
    
    # 收集embedding层参数
    if hasattr(model, 'sparse_embedding'):
        embedding_params.extend(list(model.sparse_embedding.parameters()))

    # 如果模型上有额外的projection/adaptor参数（例如Hint-KD中的投影层），也一并加入embedding优化器
    if hasattr(model, 'hint_projections'):
        # hint_projections 期望为 nn.ModuleList 或 nn.Module
        try:
            embedding_params.extend(list(model.hint_projections.parameters()))
        except Exception:
            # 忽略非模块类型
            pass
    
    # 收集其他参数（稠密模型参数）
    if hasattr(model, 'dense'):
        dense_params.extend(list(model.dense.parameters()))
    
    # 创建优化器
    embedding_optimizer = optim.Adam(embedding_params, lr=lr_sparse) ###
    dense_optimizer = optim.Adam(dense_params, lr=lr_dense)
    
    if logger:
        logger.info(f"Created separate optimizers:")
        logger.info(f"  - Adam for {len(embedding_params)} embedding parameters (lr={lr_sparse})")
        logger.info(f"  - Adam for {len(dense_params)} dense model parameters (lr={lr_dense})")
    
    return embedding_optimizer, dense_optimizer


def evaluate_model(model, loader, device):
    """Evaluates model performance, returning AUC and LogLoss."""
    model.eval()
    all_labels, all_preds = [], []
    
    try:
        with torch.no_grad():
            for x_num, x_cat, y_true in loader:
                x_num, x_cat, y_true = x_num.to(device, non_blocking=True), x_cat.to(device, non_blocking=True), y_true.to(device, non_blocking=True)
                logits = model(x_num, x_cat)
                preds = torch.sigmoid(logits)
                all_labels.extend(y_true.squeeze().tolist())
                all_preds.extend(preds.cpu().squeeze().tolist())

        if not all_labels or not all_preds:
            raise ValueError("No predictions generated during evaluation")

        auc = roc_auc_score(all_labels, all_preds)
        loss = log_loss(all_labels, all_preds)
        
        return auc*100, loss
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return 0.0, float('inf')  # Return default values on error


def plot_loss_curve(loss_history, title, save_dir, run_number=None):
    """Plots a loss curve and saves it to a file."""
    if not loss_history:
        # print(f"No loss history provided for '{title}', skipping plot.")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    filename_base = "".join(c for c in title if c.isalnum() or c in (' ', '_', '#')).rstrip()
    filename_base = filename_base.replace(' ', '_').replace('#', '')
    if run_number is not None:
        filename = f"{filename_base}_run_{run_number}.png"
    else:
        filename = f"{filename_base}.png"
    save_path = os.path.join(save_dir, filename)

    samples, losses = zip(*loss_history)
    plt.figure(figsize=(12, 6))
    plt.plot(samples, losses, label='Training Loss')
    plt.title(f'Loss Curve for {title}' + (f' (Run {run_number})' if run_number is not None else ''))
    plt.xlabel("Number of Samples Processed")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()


def distillation_loss(student_logits, teacher_logits, true_labels, temperature=4.0, alpha=0.7, method='kl'):
    """Compute distillation loss.

    Supports two methods:
      - 'kl': original implementation using soft targets + BCE-with-logits (scaled by T^2)
      - 'huber': use SmoothL1 (Huber) loss between student and teacher soft probabilities

    Args:
        student_logits: raw logits from student model
        teacher_logits: raw logits from teacher model
        true_labels: hard labels (0/1)
        temperature: distillation temperature
        alpha: weight for distillation loss vs student BCE
        method: 'kl' or 'huber'
    """
    method = (method or 'kl').lower()

    if method in ('kl', 'kldiv'):
        # KL-style distillation implemented via BCE-with-logits on soft targets
        soft_targets = torch.sigmoid(teacher_logits / temperature)
        distill_loss = F.binary_cross_entropy_with_logits(student_logits / temperature, soft_targets) * (temperature * temperature)
    elif method in ('huber', 'smoothl1'):
        # Huber distillation: compare soft probabilities (after temperature scaling)
        teacher_probs = torch.sigmoid(teacher_logits / temperature)
        student_probs = torch.sigmoid(student_logits / temperature)
        huber = nn.SmoothL1Loss(reduction='mean')
        distill_loss = huber(student_probs, teacher_probs) * (temperature * temperature)
    else:
        raise ValueError(f"Unknown distillation method: {method}")

    # Student BCE loss to hard labels
    student_loss = F.binary_cross_entropy_with_logits(student_logits, true_labels.float())
    total_loss = alpha * distill_loss + (1 - alpha) * student_loss
    return total_loss

def compute_mapping_and_transform(teacher_w: torch.Tensor, student_dim: int):
    """基于正交随机映射将 teacher embedding 转换为 student 维度。
    返回 (new_student_w, reconstruction_rel_error)
    """
    with torch.no_grad():
        vocab_size, teacher_dim = teacher_w.shape
        if teacher_dim == student_dim:
            # 直接拷贝
            return teacher_w.clone(), torch.tensor(0.0, device=teacher_w.device)

        high_dim = max(teacher_dim, student_dim)
        low_dim = min(teacher_dim, student_dim)

        if student_dim > teacher_dim:
            # 构造随机正交矩阵 Q (high_dim x high_dim)
            A = torch.randn(high_dim, high_dim, device=teacher_w.device)
            # 使用 torch.linalg.qr 获得正交矩阵
            Q, _ = torch.linalg.qr(A)  # Q 正交
            W = Q[:low_dim, :]  # (low_dim x high_dim)

            # 升维: teacher_dim = low_dim, student_dim = high_dim
            # teacher_w: (V, low_dim)  W: (low_dim x high_dim)
            new_student_w = teacher_w @ W  # (V, high_dim)
            # 重建 teacher 以估计误差: (V, high_dim) @ W.T -> (V, low_dim)
            teacher_rec = new_student_w @ W.T
            rec_err = (teacher_w - teacher_rec).norm() / (teacher_w.norm() + 1e-12)
        else:
            # # 降维: teacher_dim = high_dim, student_dim = low_dim
            # # teacher_w: (V, high_dim) W: (low_dim x high_dim)
            # new_student_w = teacher_w @ W.T  # (V, low_dim)
            # # 投影回高维估计丢失: (V, low_dim) @ W -> (V, high_dim)
            # teacher_proj = new_student_w @ W
            # rec_err = (teacher_w - teacher_proj).norm() / (teacher_w.norm() + 1e-12)

            # 降维: 使用 PCA 将 teacher_w 从 high_dim -> low_dim
            # teacher_w: (V, high_dim)
            X = teacher_w
            mean = X.mean(dim=0, keepdim=True)
            Xc = X - mean
            q = min(low_dim, Xc.shape[0], Xc.shape[1])
            if q == 0:
                # 退化情形：无法进行PCA，返回零映射
                new_student_w = torch.zeros(Xc.shape[0], low_dim, device=X.device, dtype=X.dtype)
                teacher_proj = mean.expand_as(X)
                rec_err = (X - teacher_proj).norm() / (X.norm() + 1e-12)
            else:
                try:
                    # 使用随机化 PCA
                    U, S, V = torch.pca_lowrank(Xc, q=q, center=False)
                    V_k = V[:, :q]  # (high_dim, q)
                    Y_low = Xc @ V_k  # (V, q)
                    # 若student维度 > q，零填充
                    if q < low_dim:
                        pad = torch.zeros(Xc.shape[0], low_dim - q, device=X.device, dtype=X.dtype)
                        new_student_w = torch.cat([Y_low, pad], dim=1)
                    else:
                        new_student_w = Y_low  # (V, low_dim)
                    # 重建评估
                    teacher_proj = (Y_low @ V_k.T) + mean  # (V, high_dim)
                    rec_err = (X - teacher_proj).norm() / (X.norm() + 1e-12)
                except RuntimeError:
                    # 回退：随机正交投影
                    print("PCA失败，使用随机正交投影回退")
                    A_fallback = torch.randn(high_dim, high_dim, device=X.device)
                    Q_fb, _ = torch.linalg.qr(A_fallback)
                    V_k = Q_fb[:, :q]
                    Y_low = Xc @ V_k
                    if q < low_dim:
                        pad = torch.zeros(Xc.shape[0], low_dim - q, device=X.device, dtype=X.dtype)
                        new_student_w = torch.cat([Y_low, pad], dim=1)
                    else:
                        new_student_w = Y_low
                    teacher_proj = (Y_low @ V_k.T) + mean
                    rec_err = (X - teacher_proj).norm() / (X.norm() + 1e-12)

        return new_student_w, rec_err


class TSKD_Framework:
    def __init__(self, student_model, teacher_model, config, device, teacher_optimizer_states, num_cols, cat_cols, vocab):
        self.student_model = student_model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.config = config
        self.device = device
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.vocab = vocab
        # 蒸馏温度（可由 config.yaml 配置的 distillation.temperature 或 CLI --temperature 覆盖）
        self.kd_temperature = float(self.config.get('temperature', 4.0))
        # 如果配置启用了 Hint-KD，则为每个 field 创建 projection 层，将 student embedding 映射到 teacher embedding 维度
        self.hint_projections = None
        if self.config.get('use_hint_kd', False):
            student_dim = getattr(self.student_model, 'embedding_dim', None)
            teacher_dim = getattr(self.teacher_model, 'embedding_dim', None)
            if student_dim is None or teacher_dim is None:
                raise ValueError('Models must expose embedding_dim for Hint-KD')
            # 为每个field创建线性投影（不带bias）
            projs = nn.ModuleList([
                nn.Linear(student_dim, teacher_dim, bias=False) for _ in range(self.student_model.num_fields)
            ])
            # 将投影放到目标device
            projs.to(self.device)
            self.hint_projections = projs
            # 为了让 create_separate_optimizers 能检测到 projection 参数，我们暂时把它挂到 student_model 上
            setattr(self.student_model, 'hint_projections', self.hint_projections)

        # 学生模型优化器（如果有 hint_projections，它们会被包含进 embedding optimizer）
        self.student_embedding_optimizer, self.student_dense_optimizer = create_separate_optimizers(
            self.student_model, config['lr_sparse'], config['lr_dense']
        )
        # 教师模型优化器
        self.teacher_embedding_optimizer, self.teacher_dense_optimizer = create_separate_optimizers(
            self.teacher_model, config['lr_sparse'], config['lr_dense']
        )
        
        # 如果提供了教师模型优化器状态，则恢复它们
        if teacher_optimizer_states is not None:
            self.teacher_embedding_optimizer.load_state_dict(teacher_optimizer_states['embedding_optimizer_state_dict'])
            self.teacher_dense_optimizer.load_state_dict(teacher_optimizer_states['dense_optimizer_state_dict'])
            print("Restored teacher optimizer states from checkpoint")
        else:
            print("Using fresh teacher optimizers")

        self.alpha_1b = float(self.config.get('alpha_1b', 0.7))
        self.alpha_1c = float(self.config.get('alpha_1c', 0.7))
        self.alpha_2 = float(self.config.get('alpha_2', 0.7))
        # 蒸馏损失类型：'kl' 或 'huber'（可通过 config 或 CLI --distill_loss 指定）
        self.distill_loss = str(self.config.get('distill_loss', 'kl')).lower()
        
    @staticmethod
    def _set_module_requires_grad(module, requires_grad: bool):
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _compute_hint_loss(self, student_embs, teacher_embs):
        """Compute Hint-KD MSE loss between teacher embeddings and student embeddings after projection.

        Args:
            student_embs: tensor [B, F, s_dim]
            teacher_embs: tensor [B, F, t_dim]
        Returns:
            scalar loss (already weighted by hint_lambda in config)
        """
        if not self.config.get('use_hint_kd', False) or self.hint_projections is None:
            return torch.tensor(0.0, device=self.device)

        B, num_fields, _ = student_embs.shape
        loss = torch.tensor(0.0, device=self.device)
        for f_idx, proj in enumerate(self.hint_projections):
            s_f = student_embs[:, f_idx, :]
            # project student field embedding to teacher dim
            s_proj = proj(s_f)
            t_f = teacher_embs[:, f_idx, :]
            loss = loss + F.mse_loss(s_proj, t_f, reduction='mean')

        loss = loss / max(1, len(self.hint_projections))
        return loss * float(self.config.get('hint_lambda', 1.0))

    def _compute_rkd_loss(self, student_embs, teacher_embs):
        """Compute RKD (distance + angle) with memory-safe sampling.

        Strategy:
        - Flatten per-sample features into a single vector per sample (project student into teacher space when hint_projections exist).
        - Distance: sample up to rkd_max_pairs pairs (i,j) and match normalized distances.
        - Angle: sample up to rkd_max_triplets triplets (i,j,k) and match cosine of (x_j - x_i, x_k - x_i).
        This avoids O(B^2) or O(B^4) memory.
        """
        if not self.config.get('use_rkd', False):
            return torch.tensor(0.0, device=self.device)

        B = student_embs.size(0)
        if B < 2:
            return torch.tensor(0.0, device=self.device)

        # prepare student features in teacher space
        if self.hint_projections is not None:
            s_fields = [self.hint_projections[i](student_embs[:, i, :]) for i in range(student_embs.size(1))]
            s_proj = torch.stack(s_fields, dim=1)  # [B, F, t_dim]
            s_flat = s_proj.reshape(B, -1)
        else:
            if student_embs.size(2) != teacher_embs.size(2):
                return torch.tensor(0.0, device=self.device)
            s_flat = student_embs.reshape(B, -1)

        t_flat = teacher_embs.reshape(B, -1)

        device = s_flat.device
        eps = 1e-6

        # -------------------- Distance loss (pair sampling) --------------------
        max_pairs = int(self.config.get('rkd_max_pairs', 2048))
        # sample pairs i != j
        if B * (B - 1) // 2 <= 0 or max_pairs <= 0:
            loss_dist = torch.tensor(0.0, device=device)
        else:
            # Sample indices allowing duplicates (fine for stochastic loss)
            i_idx = torch.randint(0, B, (max_pairs,), device=device)
            j_idx = torch.randint(0, B, (max_pairs,), device=device)
            neq = (i_idx != j_idx)
            if neq.sum() == 0:
                loss_dist = torch.tensor(0.0, device=device)
            else:
                i_idx = i_idx[neq]
                j_idx = j_idx[neq]
                s_diff = s_flat[i_idx] - s_flat[j_idx]
                t_diff = t_flat[i_idx] - t_flat[j_idx]
                s_d = torch.norm(s_diff, p=2, dim=1)
                t_d = torch.norm(t_diff, p=2, dim=1)
                # normalize by mean distance
                s_mean = torch.clamp(s_d.mean(), min=eps)
                t_mean = torch.clamp(t_d.mean(), min=eps)
                s_dn = s_d / s_mean
                t_dn = t_d / t_mean
                loss_dist = F.smooth_l1_loss(s_dn, t_dn)

        # -------------------- Angle loss (triplet sampling) --------------------
        max_triplets = int(self.config.get('rkd_max_triplets', 2048))
        if B < 3 or max_triplets <= 0:
            loss_angle = torch.tensor(0.0, device=device)
        else:
            a_idx = torch.randint(0, B, (max_triplets,), device=device)  # anchor
            b_idx = torch.randint(0, B, (max_triplets,), device=device)
            c_idx = torch.randint(0, B, (max_triplets,), device=device)
            # ensure all different per triplet
            mask = (a_idx != b_idx) & (a_idx != c_idx) & (b_idx != c_idx)
            if mask.sum() == 0:
                loss_angle = torch.tensor(0.0, device=device)
            else:
                a_idx = a_idx[mask]
                b_idx = b_idx[mask]
                c_idx = c_idx[mask]
                s_ab = s_flat[b_idx] - s_flat[a_idx]
                s_ac = s_flat[c_idx] - s_flat[a_idx]
                t_ab = t_flat[b_idx] - t_flat[a_idx]
                t_ac = t_flat[c_idx] - t_flat[a_idx]
                # cosine between the two vectors
                s_cos = F.cosine_similarity(s_ab, s_ac, dim=1, eps=eps)
                t_cos = F.cosine_similarity(t_ab, t_ac, dim=1, eps=eps)
                loss_angle = F.smooth_l1_loss(s_cos, t_cos)

        w_dist = float(self.config.get('rkd_w_dist', 25.0))
        w_angle = float(self.config.get('rkd_w_angle', 50.0))
        loss = w_dist * loss_dist + w_angle * loss_angle
        return loss * float(self.config.get('rkd_lambda', 1.0))
    
    def create_rep_loader(self, df_offline, offline_loader, sample_selection, run_idx, logger):
        """创建代表性样本loader"""

        # 针对流式 SplitHandle 的适配
        if isinstance(df_offline, SplitHandle):
            if sample_selection in ['random', 'temporal', 'biased_sample', 'combined']:
                # 使用全局采样 collate，实现真正的复杂采样
                # 允许从配置覆盖索引目录
                # 处理后的数据目录在项目根的 {dataset}_processed_data 下
                index_dir = os.path.join('..', self.config['dataset'] + "_processed_data", "stream_index")
                idx_mgr = StreamingIndexManager(index_dir)
                collate = StreamingSamplerCollate(
                    handle=df_offline,
                    index_mgr=idx_mgr,
                    split_name='offline',
                    batch_size=self.config['batch_size'],
                    strategy=sample_selection,
                    target_pos_rate=self.config.get('target_pos_rate', 0.5),
                    seed=42 + run_idx,
                )
                # DataLoader 用一个虚拟数据集，仅触发 batch 次数
                dummy = _DummyIndexDataset(len(df_offline))
                rep_loader = DataLoader(dummy, batch_size=self.config['batch_size'], shuffle=False, num_workers=0, collate_fn=collate)
                if logger:
                    logger.info(f"Created streaming advanced sampler for method: {sample_selection}")
            elif sample_selection == 'all':
                rep_loader = offline_loader
            elif sample_selection == 'all_shuffle':
                ds = StreamingHDF5IndexDataset(df_offline)
                rep_loader = DataLoader(ds, batch_size=self.config['batch_size'], shuffle=True, num_workers=0, collate_fn=ds.collate_fn)
            elif sample_selection == 'recent':
                length = int(self.config['sample_ratio'] * len(df_offline))
                ds = StreamingHDF5IndexDataset(df_offline, start_offset=0, length=length)
                rep_loader = DataLoader(ds, batch_size=self.config['batch_size'], shuffle=False, num_workers=0, collate_fn=ds.collate_fn)
            else:
                # 回退
                if logger:
                    logger.warning(f"Sample selection '{sample_selection}' not recognized; falling back to 'all'.")
                rep_loader = offline_loader
        else:
            # 根据sample_selection方法创建rep_loader
            if sample_selection == 'all':
                rep_loader = offline_loader  # 直接使用offline_loader
            elif sample_selection == 'all_shuffle':
                # 创建shuffle版本的loader
                rep_loader = DataLoader(MyDataset(df_offline, self.num_cols, self.cat_cols, self.vocab, self.config['dataset']), 
                                        batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'])
            elif sample_selection == 'recent':
                rep_loader = DataLoader(MyDataset(df_offline[:int(self.config['sample_ratio']*len(df_offline))], self.num_cols, self.cat_cols, self.vocab, self.config['dataset']), 
                                        batch_size=self.config['batch_size'], num_workers=self.config['num_workers'])
            else:
                # 使用动态采样Dataset
                rep_dataset = DynamicSamplingDataset(df_offline, self.num_cols, self.cat_cols, self.vocab, self.config['dataset'], 
                                                     sample_selection, self.config['sample_ratio'], self.config['target_pos_rate'], 42+run_idx)
                rep_loader = DataLoader(rep_dataset, batch_size=int(self.config['batch_size']/self.config['sample_ratio']), 
                                        shuffle=False, num_workers=self.config['num_workers'], collate_fn=rep_dataset.collate_fn)
                
                if logger:
                    logger.info(f"Created dynamic sampling rep_loader with method: {sample_selection}")
        
        return rep_loader


    def stage1(self, df_offline, offline_loader, logger=None, plot_dir=None, run_number=None, student_info=None, eval_loader=None, progress_log_config=None, time_auc_log=None):
        # Phase A: Sparse embedding InfoNCE distillation (update embeddings only)
        self.student_model.train()
        self.teacher_model.eval()

        s1_dense_epochs = self.config['s1_dense_epochs']
        s1_joint_epochs = self.config['s1_joint_epochs']

        d_rep_loader = self.create_rep_loader(df_offline, offline_loader, self.config['sample_selection'], run_number, logger)

        # 进度评估配置
        progress_enabled = bool(progress_log_config.get('enabled', False)) if progress_log_config else False
        progress_interval = int(progress_log_config.get('interval', 100)) if progress_log_config else 100
        progress_ready = progress_enabled and (eval_loader is not None) and (time_auc_log is not None)
        s1_start = time.perf_counter()
        eval_overhead = 0.0
        s1_step = 0

        if not self.config['skip_s1a']:
            # 原有方法：一次性正交线性映射（无梯度）
            loss_history_a = []  # 记录重建相对误差作为伪 loss。
            samples_processed = 0

            for field_idx in tqdm(range(self.student_model.num_fields), desc="Stage 1A (Orthogonal Mapping)"):
                teacher_layer = self.teacher_model.sparse_embedding.embedding_layers[field_idx]
                student_layer = self.student_model.sparse_embedding.embedding_layers[field_idx]

                teacher_weight = teacher_layer.weight.data  # (vocab, t_dim)
                student_dim = student_layer.weight.data.shape[1]

                new_student_weight, rec_err = compute_mapping_and_transform(teacher_weight, student_dim)

                # 赋值给 student embedding
                with torch.no_grad():
                    if student_layer.weight.data.shape != new_student_weight.shape:
                        # 安全检查
                        raise ValueError(f"Shape mismatch when assigning mapped embedding for field {field_idx}: "
                                            f"expected {student_layer.weight.data.shape}, got {new_student_weight.shape}")
                    student_layer.weight.data.copy_(new_student_weight)

                samples_processed += 1
                loss_history_a.append((samples_processed, rec_err.item()))

            # 绘图：曲线代表各 field 的重建相对误差
            plot_loss_curve(loss_history_a, f"Student (S1A-OrthoMap)_{student_info}", plot_dir, run_number)

        if not self.config['skip_s1b']:
            loss_history_b = []
            samples_processed = 0
            # Phase B: KD on D_rep updating ONLY dense model parameters (freeze embeddings)
            self._set_module_requires_grad(self.student_model.sparse_embedding, False)
           
            for epoch in range(s1_dense_epochs):
                # 使用原始D_rep数据训练
                for x_num, x_cat, y_true in tqdm(d_rep_loader, desc=f"Stage 1B (Dense Model) Epoch {epoch+1}"):
                    x_num, x_cat, y_true = x_num.to(self.device), x_cat.to(self.device), y_true.to(self.device)
                    self.student_dense_optimizer.zero_grad()
                    with torch.no_grad():
                        teacher_logits = self.teacher_model(x_num, x_cat)
                    student_logits = self.student_model(x_num, x_cat)
                    loss = distillation_loss(student_logits, teacher_logits, y_true, temperature=self.kd_temperature, alpha=self.alpha_1b, method=self.distill_loss)
                    loss.backward()
                    self.student_dense_optimizer.step()

                    samples_processed += len(y_true)
                    loss_history_b.append((samples_processed, loss.item()))        
                    s1_step += 1
                    if progress_ready and (s1_step % progress_interval == 0):
                        t_eval0 = time.perf_counter()
                        auc, lss = evaluate_model(self.student_model, eval_loader, self.device)
                        t_eval = time.perf_counter() - t_eval0
                        eval_overhead += t_eval
                        elapsed_no_eval = time.perf_counter() - s1_start - eval_overhead
                        time_auc_log['time_list'].append(float(elapsed_no_eval))
                        time_auc_log['auc_list'].append(float(auc))
                        time_auc_log['logloss_list'].append(float(lss))
                        if logger:
                            logger.info(f"[S1 Progress] step={s1_step}, time_no_eval={elapsed_no_eval:.2f}s, AUC={auc:.2f}, LogLoss={lss:.4f}")
                    
            plot_loss_curve(loss_history_b, f"Student (S1B)_{student_info}", plot_dir, run_number)

        if not self.config['skip_s1c']:
            loss_history_c = []
            samples_processed = 0
            # Phase C: KD on D_rep jointly updating embeddings + dense
            self._set_module_requires_grad(self.student_model.sparse_embedding, True)
            self._set_module_requires_grad(self.student_model.dense, True)
            
            # Stage 1C 稀疏 embedding 的 L2 正则项系数（对本批次被访问到的行做 L2）
            s1c_sparse_l2 = self.config['s1c_sparse_l2']
            # 以 Stage 1C 开始时的 sparse embedding 作为“目标”，进行L2靠拢: ||theta_S - theta_S^{init}||^2
            if s1c_sparse_l2 > 0.0:
                s1c_init_sparse_weights = [
                    emb_layer.weight.detach().clone()
                    for emb_layer in self.student_model.sparse_embedding.embedding_layers
                ]
            
            # 创建分离的优化器用于联合训练
            for epoch in range(s1_joint_epochs):
                for x_num, x_cat, y_true in tqdm(d_rep_loader, desc=f"Stage 1C (Joint) Epoch {epoch+1}"):
                    x_num, x_cat, y_true = x_num.to(self.device), x_cat.to(self.device), y_true.to(self.device)
                    
                    # 清零梯度
                    self.student_embedding_optimizer.zero_grad()
                    self.student_dense_optimizer.zero_grad()
                    
                    with torch.no_grad():
                        teacher_logits = self.teacher_model(x_num, x_cat)
                        # teacher embeddings for hint/rkd (no grad)
                        teacher_embs = self.teacher_model.sparse_embedding(x_cat)
                    student_logits = self.student_model(x_num, x_cat)
                    # student embeddings
                    student_embs = self.student_model.sparse_embedding(x_cat)
                    loss = distillation_loss(student_logits, teacher_logits, y_true, temperature=self.kd_temperature, alpha=self.alpha_1c, method=self.distill_loss)

                    # Hint-KD and RKD auxiliary losses
                    hint_loss = self._compute_hint_loss(student_embs, teacher_embs)
                    rkd_loss = self._compute_rkd_loss(student_embs, teacher_embs)
                    loss = loss + hint_loss + rkd_loss

                    # 仅在Stage 1C为稀疏embedding添加 L2 正则（只对本批用到的行，计算更高效）
                    if s1c_sparse_l2 > 0.0:
                        l2_penalty = torch.tensor(0.0, device=self.device)
                        # x_cat: [batch, num_fields]
                        for field_idx, emb_layer in enumerate(self.student_model.sparse_embedding.embedding_layers):
                            ids = x_cat[:, field_idx].reshape(-1)
                            if ids.numel() == 0:
                                continue
                            uniq = torch.unique(ids)
                            # 选取本批次涉及到的 embedding 行
                            w_rows = emb_layer.weight.index_select(0, uniq)
                            w_init_rows = s1c_init_sparse_weights[field_idx].index_select(0, uniq)
                            # 正则化项：与Stage1C初始权重的L2距离平方（使用被访问的行）
                            diff = (w_rows - w_init_rows)
                            l2_penalty = l2_penalty + diff.pow(2).sum()
                        loss = loss + s1c_sparse_l2 * l2_penalty
                    loss.backward()
                    
                    # 更新参数
                    self.student_embedding_optimizer.step()
                    self.student_dense_optimizer.step()

                    samples_processed += len(y_true)
                    loss_history_c.append((samples_processed, loss.item()))
                    s1_step += 1
                    if progress_ready and (s1_step % progress_interval == 0):
                        t_eval0 = time.perf_counter()
                        auc, lss = evaluate_model(self.student_model, eval_loader, self.device)
                        t_eval = time.perf_counter() - t_eval0
                        eval_overhead += t_eval
                        elapsed_no_eval = time.perf_counter() - s1_start - eval_overhead
                        time_auc_log['time_list'].append(float(elapsed_no_eval))
                        time_auc_log['auc_list'].append(float(auc))
                        time_auc_log['logloss_list'].append(float(lss))
                        if logger:
                            logger.info(f"[S1 Progress] step={s1_step}, time_no_eval={elapsed_no_eval:.2f}s, AUC={auc:.2f}, LogLoss={lss:.4f}")

            plot_loss_curve(loss_history_c, f"Student (S1C)_{student_info}", plot_dir, run_number)

        # 最后一条记录（S1 结束后也需要记录）
        if progress_ready and s1_step > 0:
            t_eval0 = time.perf_counter()
            auc, lss = evaluate_model(self.student_model, eval_loader, self.device)
            t_eval = time.perf_counter() - t_eval0
            eval_overhead += t_eval
            elapsed_no_eval = time.perf_counter() - s1_start - eval_overhead
            time_auc_log['time_list'].append(float(elapsed_no_eval))
            time_auc_log['auc_list'].append(float(auc))
            time_auc_log['logloss_list'].append(float(lss))
            if logger:
                logger.info(f"[S1 Final] step={s1_step}, time_no_eval={elapsed_no_eval:.2f}s, AUC={auc:.2f}, LogLoss={lss:.4f}")

        # Ensure embeddings are unfrozen for later stages
        self._set_module_requires_grad(self.student_model.sparse_embedding, True)

        # 返回 Stage 1 纯训练时间（不含评估）
        return time.perf_counter() - s1_start - eval_overhead

    def stage2(self, d_online_loader, replay_dataset=None, logger=None, plot_dir=None, run_number=None, student_info=None, teacher_info=None, eval_loader=None, progress_log_config=None, time_auc_log=None, time_offset: float = 0.0):
        self.student_model.train()
        self.teacher_model.train()

        student_loss_history = []
        teacher_loss_history = []
        samples_processed = 0
        
        # 计算总步数
        total_steps = len(d_online_loader)
        logger.info(f"Stage 2总步数: {total_steps}")

        # 进度评估配置
        progress_enabled = bool(progress_log_config.get('enabled', False)) if progress_log_config else False
        progress_interval = int(progress_log_config.get('interval', 100)) if progress_log_config else 100
        # 仅当启用且提供了评估loader与外部列表时才执行
        progress_ready = progress_enabled and (eval_loader is not None) and (time_auc_log is not None)
        # 训练计时（不包含评估时间）
        train_start_time = time.perf_counter()
        eval_overhead = 0.0
        last_step_logged = 0

        # 如果没有提供replay_dataset，使用原始逻辑
        if replay_dataset is None:
            for step, (x_num, x_cat, y_true) in enumerate(tqdm(d_online_loader, desc="Stage 2 Online Learning")):
                if self.config['stage2_update_mode'] == 'both':
                    # student_loss, teacher_loss = self.stage2_train_model_both(x_num, x_cat, y_true)
                    student_loss, teacher_loss = self.stage2_train_model_both_freq(x_num, x_cat, y_true, step % self.config['teacher_update_freq'] == 0)
                elif self.config['stage2_update_mode'] == 'student':
                    student_loss, teacher_loss = self.stage2_train_model_student(x_num, x_cat, y_true)
                samples_processed += len(y_true)
                student_loss_history.append((samples_processed, student_loss.item())) 
                teacher_loss_history.append((samples_processed, teacher_loss.item()))

                # 每 progress_interval 个 batch 记录一次时间(不含评估)和评估指标
                if progress_ready and ((step + 1) % progress_interval == 0):
                    t_eval0 = time.perf_counter()
                    auc, loss = evaluate_model(self.student_model, eval_loader, self.device)
                    t_eval = time.perf_counter() - t_eval0
                    eval_overhead += t_eval
                    elapsed_no_eval = time_offset + (time.perf_counter() - train_start_time - eval_overhead)
                    time_auc_log['time_list'].append(float(elapsed_no_eval))
                    time_auc_log['auc_list'].append(float(auc))
                    time_auc_log['logloss_list'].append(float(loss))
                    if logger:
                        logger.info(f"[Progress@{step+1}] time_no_eval={elapsed_no_eval:.2f}s, AUC={auc:.2f}, LogLoss={loss:.4f}")
                    last_step_logged = step + 1
        else:
            # 使用混合训练：同时从online和replay dataset中训练
            logger.info(f"使用ReplayDataset，每个step抽取{replay_dataset.replay_batch_size}个样本")
            
            for step, (x_num, x_cat, y_true) in enumerate(tqdm(d_online_loader, desc="Stage 2 Mixed Online+Replay Learning")):
                # 从replay dataset中采样
                replay_x_num, replay_x_cat, replay_y = replay_dataset.sample_batch()
                # replay_x_num, replay_x_cat, replay_y = replay_dataset.sample_batch_weighted()
                
                # 组合online和replay数据
                combined_x_num = torch.cat([x_num, replay_x_num], dim=0)
                combined_x_cat = torch.cat([x_cat, replay_x_cat], dim=0)
                combined_y = torch.cat([y_true, replay_y], dim=0)

                if self.config['stage2_update_mode'] == 'both':
                    # student_loss, teacher_loss = self.stage2_train_model_both(combined_x_num, combined_x_cat, combined_y)
                    # teacher只每10步更新一次
                    student_loss, teacher_loss = self.stage2_train_model_both_freq(combined_x_num, combined_x_cat, combined_y, step % self.config['teacher_update_freq'] == 0)
                elif self.config['stage2_update_mode'] == 'student':
                    student_loss, teacher_loss = self.stage2_train_model_student(combined_x_num, combined_x_cat, combined_y)

                samples_processed += len(combined_y)
                student_loss_history.append((samples_processed, student_loss.item())) 
                teacher_loss_history.append((samples_processed, teacher_loss.item()))

                # 每 progress_interval 个 batch 记录一次时间(不含评估)和评估指标
                if progress_ready and ((step + 1) % progress_interval == 0):
                    t_eval0 = time.perf_counter()
                    auc, loss = evaluate_model(self.student_model, eval_loader, self.device)
                    t_eval = time.perf_counter() - t_eval0
                    eval_overhead += t_eval
                    elapsed_no_eval = time_offset + (time.perf_counter() - train_start_time - eval_overhead)
                    time_auc_log['time_list'].append(float(elapsed_no_eval))
                    time_auc_log['auc_list'].append(float(auc))
                    time_auc_log['logloss_list'].append(float(loss))
                    if logger:
                        logger.info(f"[Progress@{step+1}] time_no_eval={elapsed_no_eval:.2f}s, AUC={auc:.2f}, LogLoss={loss:.4f}")
                    last_step_logged = step + 1
                
        plot_loss_curve(student_loss_history, f"Student (S2)_{student_info}", plot_dir, run_number)
        plot_loss_curve(teacher_loss_history, f"Teacher (S2)_{teacher_info}", plot_dir, run_number)

        # 最后一条记录（S2 结束后也需要记录）
        total_steps = len(d_online_loader)
        if progress_ready and total_steps > 0:
            if last_step_logged != total_steps:
                t_eval0 = time.perf_counter()
                auc, loss = evaluate_model(self.student_model, eval_loader, self.device)
                t_eval = time.perf_counter() - t_eval0
                eval_overhead += t_eval
                elapsed_no_eval = time_offset + (time.perf_counter() - train_start_time - eval_overhead)
                time_auc_log['time_list'].append(float(elapsed_no_eval))
                time_auc_log['auc_list'].append(float(auc))
                time_auc_log['logloss_list'].append(float(loss))
                if logger:
                    logger.info(f"[Final] time_no_eval={elapsed_no_eval:.2f}s, AUC={auc:.2f}, LogLoss={loss:.4f}")

        # 返回 Stage 2 纯训练时间（不含评估）
        return time.perf_counter() - train_start_time - eval_overhead

    
    def stage2_train_model_both_freq(self, x_num, x_cat, y_true, update_teacher=False):
        x_num, x_cat, y_true = x_num.to(self.device), x_cat.to(self.device), y_true.to(self.device)
        
        # 清零梯度
        self.student_embedding_optimizer.zero_grad()
        self.student_dense_optimizer.zero_grad()
        
        # 前向传播
        student_logits = self.student_model(x_num, x_cat)
        # student embeddings
        student_embs = self.student_model.sparse_embedding(x_cat)
        teacher_logits = self.teacher_model(x_num, x_cat)
        # teacher embeddings (for hint/rkd) - detach to avoid teacher grad
        with torch.no_grad():
            teacher_embs = self.teacher_model.sparse_embedding(x_cat)
        teacher_logits2 = teacher_logits.detach()

        student_loss = distillation_loss(student_logits, teacher_logits2, y_true, temperature=self.kd_temperature, alpha=self.alpha_2, method=self.distill_loss)
        # add auxiliary hint/rkd losses
        student_loss = student_loss + self._compute_hint_loss(student_embs, teacher_embs) + self._compute_rkd_loss(student_embs, teacher_embs)
        teacher_loss = F.binary_cross_entropy_with_logits(teacher_logits, y_true)
        
        # 反向传播
        student_loss.backward()
        teacher_loss.backward()
        
        # 更新参数
        self.student_embedding_optimizer.step()
        self.student_dense_optimizer.step()

        if update_teacher:
            self.teacher_embedding_optimizer.step()
            self.teacher_dense_optimizer.step()
            self.teacher_embedding_optimizer.zero_grad()
            self.teacher_dense_optimizer.zero_grad()
        
        return student_loss, teacher_loss

    def stage2_train_model_both(self, x_num, x_cat, y_true):
        x_num, x_cat, y_true = x_num.to(self.device), x_cat.to(self.device), y_true.to(self.device)
        
        # 清零梯度
        self.student_embedding_optimizer.zero_grad()
        self.student_dense_optimizer.zero_grad()
        self.teacher_embedding_optimizer.zero_grad()
        self.teacher_dense_optimizer.zero_grad()
        
        # 前向传播
        student_logits = self.student_model(x_num, x_cat)
        student_embs = self.student_model.sparse_embedding(x_cat)
        teacher_logits = self.teacher_model(x_num, x_cat)
        with torch.no_grad():
            teacher_embs = self.teacher_model.sparse_embedding(x_cat)
        teacher_logits2 = teacher_logits.detach()

        student_loss = distillation_loss(student_logits, teacher_logits2, y_true, temperature=self.kd_temperature, alpha=self.alpha_2, method=self.distill_loss)
        # add auxiliary hint/rkd
        student_loss = student_loss + self._compute_hint_loss(student_embs, teacher_embs) + self._compute_rkd_loss(student_embs, teacher_embs)
        teacher_loss = F.binary_cross_entropy_with_logits(teacher_logits, y_true)
        
        # 反向传播
        student_loss.backward()
        teacher_loss.backward()
        
        # 更新参数
        self.student_embedding_optimizer.step()
        self.student_dense_optimizer.step()
        self.teacher_embedding_optimizer.step()
        self.teacher_dense_optimizer.step()
        
        return student_loss, teacher_loss

    def stage2_train_model_student(self, x_num, x_cat, y_true):
        x_num, x_cat, y_true = x_num.to(self.device), x_cat.to(self.device), y_true.to(self.device)
        
        # 清零梯度
        self.student_embedding_optimizer.zero_grad()
        self.student_dense_optimizer.zero_grad()
        
        # 前向传播
        student_logits = self.student_model(x_num, x_cat)
        student_embs = self.student_model.sparse_embedding(x_cat)
        teacher_logits = self.teacher_model(x_num, x_cat)
        with torch.no_grad():
            teacher_embs = self.teacher_model.sparse_embedding(x_cat)
        teacher_logits2 = teacher_logits.detach()

        student_loss = distillation_loss(student_logits, teacher_logits2, y_true, temperature=self.kd_temperature, alpha=self.alpha_2, method=self.distill_loss)
        # add auxiliary hint/rkd
        student_loss = student_loss + self._compute_hint_loss(student_embs, teacher_embs) + self._compute_rkd_loss(student_embs, teacher_embs)
        
        # 反向传播
        student_loss.backward()
        
        # 更新参数
        self.student_embedding_optimizer.step()
        self.student_dense_optimizer.step()
        
        return student_loss, torch.tensor(0.0)

