# Data Preparation Module
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import gc
import psutil
from .dataset_stream import SplitHandle, OffsetManager


def nested_defaultdict_int():
    """A picklable helper function for creating a defaultdict of ints."""
    return defaultdict(int)

def log_memory_usage(stage_name):
    """记录内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / 1024 / 1024 / 1024
    print(f"[{stage_name}] 内存使用: {memory_gb:.2f} GB")
    return memory_gb

def force_garbage_collection():
    """强制垃圾回收"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def optimize_dtypes(df, num_cols=None, cat_cols=None, label_col='label'):
    """
    优化数据类型以减少内存使用
    
    Args:
        df (pd.DataFrame): 输入的数据帧
        num_cols (list): 数值特征列（可选）
        cat_cols (list): 类别特征列（可选）
        label_col (str): 标签列名
    
    Returns:
        pd.DataFrame: 优化后的数据帧
    """
    # 优化数值列的数据类型
    if num_cols:
        for col in num_cols:
            if col in df.columns:
                # 使用float32而不是float64
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

    # 优化类别列的数据类型
    if cat_cols:
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
    # 优化标签列
    df[label_col] = df[label_col].astype(np.int8)
    return df


def optimize_dtypes_for_hdf5(df, num_cols=None, cat_cols=None, label_col='label'):
    """
    优化数据类型以减少内存使用，但避免使用category类型以支持HDF5追加。
    数值列使用 float32；类别列使用 int32；标签使用 int8。
    """
    if num_cols:
        for col in num_cols:
            if col in df.columns:
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
    if cat_cols:
        for col in cat_cols:
            if col in df.columns:
                df.loc[:, col] = df[col].astype(np.int32)
    if label_col in df.columns:
        df.loc[:, label_col] = df[label_col].astype(np.int8)
    return df


def build_vocab_from_chunks(chunks, cat_cols, categorical_threshold=10):
    """
    从分块数据构建词汇表；每列保留出现次数>=阈值的类别，其余映射到 <UNK>=0。
    注意：此函数仅用于首次预处理时构建 vocab.pkl，流式读取阶段通常直接加载缓存。
    """
    print("构建词汇表...")
    value_counts = defaultdict(lambda: defaultdict(int))
    for chunk in chunks:
        for col in cat_cols:
            if col in chunk.columns:
                counts = chunk[col].value_counts()
                for val, cnt in counts.items():
                    value_counts[col][val] += int(cnt)

    # 统计并打印各列低频过滤信息
    total_unique_all, total_kept_all, total_filtered_all = 0, 0, 0
    print("类别频次统计与过滤概览：")
    for col in cat_cols:
        col_counts = value_counts[col]
        total_unique = len(col_counts)
        kept = sum(1 for v, c in col_counts.items() if c >= categorical_threshold and v != '<UNK>')
        filtered = total_unique - kept
        total_unique_all += total_unique
        total_kept_all += kept
        total_filtered_all += filtered
        ratio = (filtered / total_unique * 100) if total_unique > 0 else 0.0
        print(f"  - 列 {col}: 总类别 {total_unique:,}，保留 {kept:,}，过滤 {filtered:,} ({ratio:.2f}%)，阈值 {categorical_threshold}")
    if total_unique_all > 0:
        overall_ratio = total_filtered_all / total_unique_all * 100
        print(f"合计: 总类别 {total_unique_all:,}，保留 {total_kept_all:,}，过滤 {total_filtered_all:,} ({overall_ratio:.2f}%)")

    vocab = defaultdict(nested_defaultdict_int)
    for col in cat_cols:
        vocab[col]['<UNK>'] = 0
        idx = 1
        for val, cnt in value_counts[col].items():
            if cnt >= categorical_threshold and val != '<UNK>':
                vocab[col][val] = idx
                idx += 1
    return vocab


def split_data_by_time(df_full, train_ratio=0.4, offline_ratio=0.4, online_ratio=0.1):
    """
    按时间顺序严格划分数据集，保持原始顺序不打乱。
    返回 df_hist, df_offline, df_online, df_test。
    """
    print("按时间顺序划分数据集...")
    log_memory_usage("开始时间划分")
    total_size = len(df_full)
    hist_size = int(total_size * train_ratio)
    offline_size = int(total_size * offline_ratio)
    online_size = int(total_size * online_ratio)
    df_hist = df_full.iloc[:hist_size]
    df_offline = df_full.iloc[hist_size:(hist_size + offline_size)]
    df_online = df_full.iloc[(hist_size + offline_size):(hist_size + offline_size + online_size)]
    df_test = df_full.iloc[(hist_size + offline_size + online_size):]
    log_memory_usage("时间划分完成")
    return df_hist, df_offline, df_online, df_test


class DataProcessor:
    def __init__(self, dataset_name, full_in_memory: bool = False):
        self.dataset_name = dataset_name
        self.full_in_memory = bool(full_in_memory)
        self.chunk_size = 100000

        if self.dataset_name == 'avazu':
            self.data_path = os.path.join("avazu_data", "train")
            self.label_col = 'click'
            cols = [
                'id', 'click', 'hour', 'C1', 'banner_pos', 'site_id',
                'site_domain', 'site_category', 'app_id', 'app_domain',
                'app_category', 'device_id', 'device_ip', 'device_model',
                'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17',
                'C18', 'C19', 'C20', 'C21'
            ]
            self.cat_cols = [col for col in cols if col not in ['id', 'click']]
            self.num_cols = []
            self.categorical_threshold = 5
            self.sep = ','
            self.has_header = True
            self.use_streaming = False
            self.train_ratio, self.offline_ratio, self.online_ratio = 0.4, 0.4, 0.1

        elif self.dataset_name == 'criteo':
            self.data_path = os.path.join("criteo_data", "train.txt")
            self.label_col = 'label'
            self.num_cols = [f'I{i}' for i in range(1, 14)]
            self.cat_cols = [f'C{i}' for i in range(1, 27)]
            self.categorical_threshold = 10
            self.sep = '\t'
            self.has_header = False
            self.use_streaming = False
            self.train_ratio, self.offline_ratio, self.online_ratio = 0.4, 0.4, 0.1

        elif self.dataset_name == 'criteo_ctr_full':
            self.data_path = os.path.join("criteo1T", "criteo1T_sample_3pct.txt")
            self.label_col = 'label'
            self.num_cols = [f'I{i}' for i in range(1, 14)]
            self.cat_cols = [f'C{i}' for i in range(1, 27)]
            self.categorical_threshold = 30
            self.sep = '\t'
            self.has_header = False
            # 全量内存模式下不使用流式
            self.use_streaming = not self.full_in_memory
            self.train_ratio, self.offline_ratio, self.online_ratio = 0.5, 0.4, 0.05

        else:
            raise ValueError(
                "Unsupported dataset name. Choose 'avazu' or 'criteo'.")

        # --- 数据加载和预处理 ---
        self.processed_data_dir = os.path.join(f"{self.dataset_name}_processed_data")
        os.makedirs(self.processed_data_dir, exist_ok=True)

        # 根据是否使用流式处理选择文件格式
        if self.use_streaming:
            self.hist_path = os.path.join(self.processed_data_dir, "hist.h5")
            self.offline_path = os.path.join(self.processed_data_dir, "offline.h5")
            self.online_path = os.path.join(self.processed_data_dir, "online.h5")
            self.test_path = os.path.join(self.processed_data_dir, "test.h5")
        else:
            # 非流式：对于 criteo_ctr_full 的全量内存模式，直接使用 parquet；其他数据集保持 pkl
            if self.dataset_name == 'criteo_ctr_full':
                self.hist_path = os.path.join(self.processed_data_dir, "hist.parquet")
                self.offline_path = os.path.join(self.processed_data_dir, "offline.parquet")
                self.online_path = os.path.join(self.processed_data_dir, "online.parquet")
                self.test_path = os.path.join(self.processed_data_dir, "test.parquet")
            else:
                self.hist_path = os.path.join(self.processed_data_dir, "hist.pkl")
                self.offline_path = os.path.join(self.processed_data_dir, "offline.pkl")
                self.online_path = os.path.join(self.processed_data_dir, "online.pkl")
                self.test_path = os.path.join(self.processed_data_dir, "test.pkl")

        self.vocab_path = os.path.join(self.processed_data_dir, "vocab.pkl")
        self.cols = [self.label_col] + self.num_cols + self.cat_cols
        log_memory_usage("开始处理")

    def load_data_in_chunks(self):
        # 获取文件总行数用于进度条
        total_lines = sum(1 for _ in open(self.data_path, 'r'))
        print(f"总行数: {total_lines:,}")

        # 根据是否有表头调整读取参数
        if self.has_header:
            chunk_iter = pd.read_csv(self.data_path, sep=self.sep, header=0, names=self.cols, chunksize=self.chunk_size, dtype=str)  # 先以字符串读取，后续优化类型
        else:
            chunk_iter = pd.read_csv(self.data_path, sep=self.sep, header=None, names=self.cols, chunksize=self.chunk_size, dtype=str)  # 先以字符串读取，后续优化类型

        for i, chunk in enumerate(
                tqdm(chunk_iter,
                     total=total_lines // self.chunk_size + 1,
                     desc="加载数据块")):
            # log_memory_usage(f"处理块 {i+1}")
            yield chunk
            force_garbage_collection()

    def preprocess_chunk(self, df, vocab):
        # 确保标签列存在且为整数类型
        df.dropna(subset=[self.label_col], inplace=True)
        df[self.label_col] = pd.to_numeric(df[self.label_col], errors='coerce')
        # 处理转换失败的情况，用0填充
        df[self.label_col] = df[self.label_col].fillna(0).astype(np.int8)

        # 处理数值特征：先转换为数值类型，再填充缺失值
        if self.num_cols:
            for col in self.num_cols:
                if col in df.columns:
                    # 转换为数值类型，无法转换的设为NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # 填充缺失值为0
                    df[col] = df[col].fillna(0)
                    # 根据 f(x) = log2(x) for x > 2 的规则进行变换
                    df[col] = df[col].map(lambda x: np.log2(x) if x > 2 else x)

        # 处理类别特征
        for col in self.cat_cols:
            if col in df.columns:
                # 填充缺失值
                df[col] = df[col].fillna(0)
                # 替换稀有值
                df[col] = df[col].map(lambda x: vocab[col].get(x, 0))

        # 优化数据类型，但不使用category类型以避免HDF5追加问题
        df = optimize_dtypes_for_hdf5(df, self.num_cols, self.cat_cols, self.label_col)

        return df

    def save_processed_data(self, df_hist, df_offline, df_online, df_test):
        """
        保存处理好的数据
        """
        log_memory_usage("开始保存数据")
        # 非流式模式使用pkl格式
        df_hist.to_pickle(self.hist_path)
        df_offline.to_pickle(self.offline_path)
        df_online.to_pickle(self.online_path)
        df_test.to_pickle(self.test_path)
        log_memory_usage("保存缓存后")

    def process_data_streaming_with_split(self, vocab):
        """
        流式处理数据并直接按时间顺序划分到不同文件
        """
        log_memory_usage("开始流式处理并划分数据")

        # 计算划分比例
        total_lines = sum(1 for _ in open(self.data_path, 'r'))
        hist_size = int(total_lines * self.train_ratio)
        offline_size = int(total_lines * self.offline_ratio)
        online_size = int(total_lines * self.online_ratio)

        print(f"总行数: {total_lines:,}")
        print(f"历史训练集: {hist_size:,} 行")
        print(f"离线学习集: {offline_size:,} 行")
        print(f"在线学习集: {online_size:,} 行")
        print(
            f"测试集: {total_lines - hist_size - offline_size - online_size:,} 行")

        write_hist, write_offline, write_online, write_test = False, False, False, False
        if not os.path.exists(self.hist_path):
            write_hist = True
            pd.DataFrame(columns=self.cols).to_hdf(self.hist_path, key='data', mode='w', format='table')
        if not os.path.exists(self.offline_path):
            write_offline = True
            pd.DataFrame(columns=self.cols).to_hdf(self.offline_path, key='data', mode='w', format='table')
        if not os.path.exists(self.online_path):
            write_online = True
            pd.DataFrame(columns=self.cols).to_hdf(self.online_path, key='data', mode='w', format='table')
        if not os.path.exists(self.test_path):
            write_test = True
            pd.DataFrame(columns=self.cols).to_hdf(self.test_path, key='data', mode='w', format='table')

        print("流式处理并划分数据...")
        chunks = self.load_data_in_chunks()
        current_line = 0

        for i, chunk in enumerate(chunks):
            chunk_size = len(chunk)

            # 根据当前行数确定数据块应该写入哪个文件
            if current_line < hist_size:
                # 历史训练集
                end_line = min(current_line + chunk_size, hist_size)
                if end_line > current_line:
                    if write_hist:
                        write_chunk = chunk.iloc[:end_line - current_line]
                        write_chunk = self.preprocess_chunk(write_chunk, vocab)
                        write_chunk.to_hdf(self.hist_path, key='data', mode='a', format='table', append=True)
                    if write_offline and end_line < current_line + chunk_size:
                        # 剩余部分属于离线学习集
                        remaining_chunk = chunk.iloc[end_line - current_line:]
                        remaining_chunk = self.preprocess_chunk(
                            remaining_chunk, vocab)
                        remaining_chunk.to_hdf(self.offline_path, key='data', mode='a', format='table', append=True)
            elif current_line < hist_size + offline_size:
                # 离线学习集
                end_line = min(current_line + chunk_size,
                               hist_size + offline_size)
                if end_line > current_line:
                    if write_offline:
                        write_chunk = chunk.iloc[:end_line - current_line]
                        write_chunk = self.preprocess_chunk(write_chunk, vocab)
                        write_chunk.to_hdf(self.offline_path, key='data', mode='a', format='table', append=True)
                    if write_online and end_line < current_line + chunk_size:
                        # 剩余部分属于在线学习集
                        remaining_chunk = chunk.iloc[end_line - current_line:]
                        remaining_chunk = self.preprocess_chunk(
                            remaining_chunk, vocab)
                        remaining_chunk.to_hdf(self.online_path, key='data', mode='a', format='table', append=True)
            elif current_line < hist_size + offline_size + online_size:
                # 在线学习集
                end_line = min(current_line + chunk_size,
                               hist_size + offline_size + online_size)
                if end_line > current_line:
                    if write_online:
                        write_chunk = chunk.iloc[:end_line - current_line]
                        write_chunk = self.preprocess_chunk(write_chunk, vocab)
                        write_chunk.to_hdf(self.online_path, key='data', mode='a', format='table', append=True)
                    if write_test and end_line < current_line + chunk_size:
                        # 剩余部分属于测试集
                        remaining_chunk = chunk.iloc[end_line - current_line:]
                        remaining_chunk = self.preprocess_chunk(
                            remaining_chunk, vocab)
                        remaining_chunk.to_hdf(self.test_path, key='data', mode='a', format='table', append=True)
            else:
                # 测试集
                if write_test:
                    chunk = self.preprocess_chunk(chunk, vocab)
                    chunk.to_hdf(self.test_path, key='data', mode='a', format='table', append=True)

            current_line += chunk_size
            force_garbage_collection()

        log_memory_usage("流式处理并划分完成")

    def process_data(self):
        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, 'rb') as f:
                vocab = pickle.load(f)
        else:
            log_memory_usage("开始处理前")
            chunks = self.load_data_in_chunks()
            vocab = build_vocab_from_chunks(
                chunks,
                self.cat_cols,
                categorical_threshold=self.categorical_threshold)
            log_memory_usage("构建词汇表后")
            with open(self.vocab_path, 'wb') as f:
                pickle.dump(vocab, f)

        if self.use_streaming:
            if not all(
                    os.path.exists(p) for p in [
                        self.hist_path, self.offline_path, self.online_path,
                        self.test_path
                    ]):
                print("\n--- 未找到缓存，开始完整的数据处理 ---")
                self.process_data_streaming_with_split(vocab)

            # 对于 criteo_ctr_full，返回轻量句柄，按批次读取，避免一次性加载
            if self.dataset_name == 'criteo_ctr_full':
                print("\n--- 准备偏移量信息（Lazy/Streaming） ---")
                om = OffsetManager(self.processed_data_dir)
                meta = om.build_or_load({
                    'hist': self.hist_path,
                    'offline': self.offline_path,
                    'online': self.online_path,
                    'test': self.test_path,
                })
                df_hist = SplitHandle(meta['hist']['path'], meta['hist']['nrows'], self.dataset_name, self.num_cols, self.cat_cols, self.label_col)
                df_offline = SplitHandle(meta['offline']['path'], meta['offline']['nrows'], self.dataset_name, self.num_cols, self.cat_cols, self.label_col)
                df_online = SplitHandle(meta['online']['path'], meta['online']['nrows'], self.dataset_name, self.num_cols, self.cat_cols, self.label_col)
                df_test = SplitHandle(meta['test']['path'], meta['test']['nrows'], self.dataset_name, self.num_cols, self.cat_cols, self.label_col)
            else:
                print("\n--- 从缓存加载已处理的数据 ---")
                log_memory_usage("加载缓存前")
                # 其他数据集保持原有方式
                df_hist = pd.read_hdf(self.hist_path, 'data')
                df_offline = pd.read_hdf(self.offline_path, 'data')
                df_online = pd.read_hdf(self.online_path, 'data')
                df_test = pd.read_hdf(self.test_path, 'data')
        else:
            if all(os.path.exists(p) for p in [self.hist_path, self.offline_path, self.online_path, self.test_path]):
                # 非流式模式：根据扩展名读取 pkl 或 parquet
                ext = os.path.splitext(self.hist_path)[1].lower()
                if ext == '.parquet':
                    df_hist = pd.read_parquet(self.hist_path)
                    df_offline = pd.read_parquet(self.offline_path)
                    df_online = pd.read_parquet(self.online_path)
                    df_test = pd.read_parquet(self.test_path)
                else:
                    df_hist = pd.read_pickle(self.hist_path)
                    df_offline = pd.read_pickle(self.offline_path)
                    df_online = pd.read_pickle(self.online_path)
                    df_test = pd.read_pickle(self.test_path)
                log_memory_usage("加载缓存后")
                print("数据加载完成。")
            else:
                # 分块加载数据进行处理
                log_memory_usage("开始处理前")
                chunks = self.load_data_in_chunks()
                all_processed_chunks = []
                for i, chunk in enumerate(chunks):
                    # 预处理数据块
                    chunk = self.preprocess_chunk(chunk, vocab)
                    # 保存处理后的数据块
                    all_processed_chunks.append(chunk)
                    # 定期清理内存
                    if i % 10 == 0:
                        force_garbage_collection()
                # 合并所有数据块，保持时间顺序
                print("合并所有数据块...")
                df_full = pd.concat(all_processed_chunks, ignore_index=True)
                del all_processed_chunks
                force_garbage_collection()

                log_memory_usage("数据合并后")
                # 按时间顺序划分数据集
                df_hist, df_offline, df_online, df_test = split_data_by_time(
                        df_full, self.train_ratio, self.offline_ratio, self.online_ratio)
                del df_full
                force_garbage_collection()
                # 保存数据
                log_memory_usage("开始保存数据")
                ext = os.path.splitext(self.hist_path)[1].lower()
                if ext == '.parquet':
                    df_hist.to_parquet(self.hist_path, index=False)
                    df_offline.to_parquet(self.offline_path, index=False)
                    df_online.to_parquet(self.online_path, index=False)
                    df_test.to_parquet(self.test_path, index=False)
                else:
                    df_hist.to_pickle(self.hist_path)
                    df_offline.to_pickle(self.offline_path)
                    df_online.to_pickle(self.online_path)
                    df_test.to_pickle(self.test_path)
                log_memory_usage("保存缓存后")

        # 汇总信息与返回（此处放在函数内部以保持作用域正确）
        print(f"\n历史训练集: {len(df_hist):,} 行")
        print(f"离线学习集: {len(df_offline):,} 行")
        print(f"在线学习集: {len(df_online):,} 行")
        print(f"测试集: {len(df_test):,} 行")

        vocab_sizes = {col: len(v) for col, v in vocab.items()}
        log_memory_usage("处理完成")

        # 返回所有需要的部分
        return df_hist, df_offline, df_online, df_test, self.num_cols, self.cat_cols, vocab, vocab_sizes


class MyDataset(Dataset):
    def __init__(self, df, numerical_cols, categorical_cols, vocab, dataset_name):
        """
        创建数据集对象

        Args:
            df (pd.DataFrame): 数据帧
            num_cols (list): 数值特征列
            cat_cols (list): 类别特征列
            vocab (dict): 词汇表
            preload (bool): 是否使用预加载方式，如果是则预处理所有数据为tensor
        
        Returns:
            Dataset: 对应的数据集对象
        """
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.vocab = vocab
        self.dataset_name = dataset_name
        # 预先计算标签，因为标签转换相对简单
        label_col = 'label' if self.dataset_name in ['criteo', 'criteo_ctr_full'] else 'click'
        labels_np = df[label_col].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
        self.labels = torch.from_numpy(labels_np)

        if numerical_cols:
            numerical_values = df[numerical_cols].to_numpy(dtype=np.float32, copy=False)
            self.numerical_data = torch.from_numpy(numerical_values)
        else:
            self.numerical_data = torch.empty((len(df), 0), dtype=torch.float32)

        # 预处理类别特征
        categorical_values = df[categorical_cols].to_numpy(dtype=np.int64, copy=False)
        self.categorical_data = torch.from_numpy(categorical_values).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.numerical_cols:
            return self.numerical_data[idx], self.categorical_data[idx], self.labels[idx]
        else:
            return torch.tensor([]), self.categorical_data[idx], self.labels[idx]


class ReplayDataset(Dataset):
    """
    动态Replay数据集，支持从df_offline中随机采样
    """
    def __init__(self, df_offline, numerical_cols, categorical_cols, vocab,
                 dataset_name, replay_batch_size):
        """
        创建Replay数据集对象
        
        Args:
            df_offline (pd.DataFrame): 离线数据帧
            numerical_cols (list): 数值特征列
            categorical_cols (list): 类别特征列
            vocab (dict): 词汇表
            dataset_name (str): 数据集名称
            replay_batch_size (int): 每次采样的样本数量
        """
        self.df_offline = df_offline
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.vocab = vocab
        self.dataset_name = dataset_name
        self.replay_batch_size = replay_batch_size

        # 确定标签列名
        self.label_col = 'label' if self.dataset_name in ['criteo', 'criteo_ctr_full'] else 'click'

        # 预计算一些信息以提高效率
        self.total_samples = len(df_offline)

    def sample_batch(self):
        """
        随机采样replay_batch_size个样本
        
        Args:
            idx: 索引（这里不使用，只是为了兼容Dataset接口）
            
        Returns:
            tuple: (numerical_data, categorical_data, labels)
        """
        # 随机采样
        sampled_indices = np.random.choice(self.total_samples,
                                           size=self.replay_batch_size,
                                           replace=True)
        sampled_df = self.df_offline.iloc[sampled_indices]

        # 转换为tensor
        if self.numerical_cols:
            numerical_np = sampled_df[self.numerical_cols].to_numpy(dtype=np.float32, copy=False)
            numerical_data = torch.from_numpy(numerical_np)
        else:
            numerical_data = torch.empty((self.replay_batch_size, 0), dtype=torch.float32)

        categorical_np = sampled_df[self.categorical_cols].to_numpy(dtype=np.int64, copy=False)
        categorical_data = torch.from_numpy(categorical_np).long()
        labels_np = sampled_df[self.label_col].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
        labels = torch.from_numpy(labels_np)

        return numerical_data, categorical_data, labels


class DynamicSamplingDataset(Dataset):
    """
    动态采样数据集，支持在每次获取batch时进行不同的采样策略
    """
    def __init__(self, df_offline, numerical_cols, categorical_cols, vocab, dataset_name,
                 sample_selection, sample_ratio, target_pos_rate=0.5, seed=42):
        """
        创建动态采样数据集对象
        
        Args:
            df_offline (pd.DataFrame): 离线数据帧
            numerical_cols (list): 数值特征列
            categorical_cols (list): 类别特征列
            vocab (dict): 词汇表
            dataset_name (str): 数据集名称
            batch_size (int): 批次大小
            sample_selection (str): 采样策略
            config (dict): 配置参数
            seed (int): 随机种子
        """
        self.df_offline = df_offline
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.vocab = vocab
        self.dataset_name = dataset_name
        self.sample_selection = sample_selection
        self.sample_ratio = sample_ratio
        self.target_pos_rate = target_pos_rate
        self.seed = seed

        # 确定标签列名
        self.label_col = 'label' if self.dataset_name in ['criteo', 'criteo_ctr_full'] else 'click'

        # 总样本数
        self.total_samples = len(df_offline)

        # 在初始化时保存标签，便于进行有偏采样
        self.labels = df_offline[self.label_col].values

        # 设置随机种子
        np.random.seed(seed)

        # 根据时间采样的全局采样概率计算
        if sample_selection in ['temporal', 'combined']:
            # 假设数据按时间顺序排列，越新的数据权重越高
            # 使用线性递增的权重
            self.temporal_weights = np.arange(1, self.total_samples + 1)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return idx

    def collate_fn(self, batch):
        """
        自定义的collate函数，在batch级别进行采样
        
        Args:
            batch: 从__getitem__返回的样本列表
            
        Returns:
            tuple: (numerical_data, categorical_data, labels) - 采样后的batch
        """
        # 从原始数据中随机采样
        if self.sample_selection == 'random':
            sampled_indices = np.random.choice(batch, size=int(len(batch) * self.sample_ratio), replace=True)
        elif self.sample_selection == 'biased_sample':
            batch_labels = self.labels[batch]
            pos_mask = batch_labels == 1
            neg_mask = batch_labels == 0
            pos_indices = np.array(batch)[pos_mask]
            neg_indices = np.array(batch)[neg_mask]
            num_pos_target = int(len(batch) * self.sample_ratio * self.target_pos_rate)
            num_neg_target = int(len(batch) * self.sample_ratio) - num_pos_target
            sampled_pos = np.random.choice(pos_indices, size=num_pos_target, replace=True)
            sampled_neg = np.random.choice(neg_indices, size=num_neg_target, replace=True)
            sampled_indices = np.concatenate([sampled_pos, sampled_neg])
        elif self.sample_selection == 'temporal':
            p = self.temporal_weights[batch] / self.temporal_weights[batch].sum()
            sampled_indices = np.random.choice(batch, size=int(len(batch) * self.sample_ratio), replace=True, p=p)
        elif self.sample_selection == 'combined':
            batch_labels = self.labels[batch]
            pos_mask = batch_labels == 1
            neg_mask = batch_labels == 0
            pos_indices = np.array(batch)[pos_mask]
            neg_indices = np.array(batch)[neg_mask]
            num_pos_target = int(len(batch) * self.sample_ratio * self.target_pos_rate)
            num_neg_target = int(len(batch) * self.sample_ratio) - num_pos_target
            pos_p = self.temporal_weights[pos_indices]
            pos_p = pos_p / pos_p.sum()
            sampled_pos = np.random.choice(pos_indices, size=num_pos_target, replace=True, p=pos_p)
            neg_p = self.temporal_weights[neg_indices]
            neg_p = neg_p / neg_p.sum()
            sampled_neg = np.random.choice(neg_indices, size=num_neg_target, replace=True, p=neg_p)
            sampled_indices = np.concatenate([sampled_pos, sampled_neg])
        else:
            raise ValueError(f"Invalid sample selection: {self.sample_selection}")

        # 获取采样的数据
        sampled_df = self.df_offline.iloc[sampled_indices]

        # 转换为tensor
        if self.numerical_cols:
            numerical_np = sampled_df[self.numerical_cols].to_numpy(dtype=np.float32, copy=False)
            numerical_data = torch.from_numpy(numerical_np)
        else:
            numerical_data = torch.empty((len(sampled_df), 0), dtype=torch.float32)

        categorical_np = sampled_df[self.categorical_cols].to_numpy(dtype=np.int64, copy=False)
        categorical_data = torch.from_numpy(categorical_np).long()
        labels_np = sampled_df[self.label_col].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
        labels = torch.from_numpy(labels_np)

        return numerical_data, categorical_data, labels

