import os
"""
HDF5 tuning for multi-process read performance:
- Disable file locking in read-only scenarios to avoid lock contention across
    DataLoader workers (Linux only). Users can override by setting the env var
    before starting the process.
"""
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import json
import time
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SplitHandle:
    """
    轻量级数据分片句柄，仅包含路径与行数，用于延迟按批次读取。
    """
    def __init__(self, path: str, nrows: int, dataset_name: str, num_cols: List[str], cat_cols: List[str], label_col: str):
        self.path = path
        self.nrows = int(nrows)
        self.dataset_name = dataset_name
        self.num_cols = list(num_cols) if num_cols else []
        self.cat_cols = list(cat_cols) if cat_cols else []
        self.label_col = label_col

    def __len__(self):
        return self.nrows


class OffsetManager:
    """
    偏移量/元信息管理器：对 HDF5 分片文件记录行数与校验信息，避免提前加载数据。
    """
    def __init__(self, processed_data_dir: str):
        self.processed_data_dir = processed_data_dir
        self.offsets_path = os.path.join(processed_data_dir, "offsets_criteo_ctr_full.json")

    def _file_info(self, path: str) -> Dict[str, Any]:
        try:
            stat = os.stat(path)
            return {"mtime": stat.st_mtime, "size": stat.st_size}
        except FileNotFoundError:
            return {"mtime": 0, "size": 0}

    def _compute_nrows(self, h5_path: str) -> int:
        try:
            with pd.HDFStore(h5_path, mode='r') as store:
                storer = store.get_storer('data')
                return int(getattr(storer, 'nrows', 0))
        except (FileNotFoundError, KeyError):
            return 0

    def build_or_load(self, paths: Dict[str, str]) -> Dict[str, Any]:
        current_meta = {k: {"path": v, "file": self._file_info(v)} for k, v in paths.items()}
        if os.path.exists(self.offsets_path):
            try:
                with open(self.offsets_path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                valid = True
                for k in paths.keys():
                    if k not in saved:
                        valid = False
                        break
                    saved_file = saved[k].get('file', {})
                    cur_file = current_meta[k]['file']
                    if abs(saved_file.get('mtime', 0) - cur_file.get('mtime', 0)) > 1e-6 or saved_file.get('size', -1) != cur_file.get('size', -2):
                        valid = False
                        break
                if valid:
                    return saved
            except Exception:
                pass
        result = {}
        for k, p in paths.items():
            nrows = self._compute_nrows(p)
            result[k] = {"path": p, "nrows": nrows, "file": current_meta[k]['file']}
        try:
            with open(self.offsets_path, 'w', encoding='utf-8') as f:
                json.dump(result, f)
        except Exception:
            pass
        return result


class StreamingHDF5IndexDataset(Dataset):
    """
    基于 HDF5 的流式索引数据集：不在初始化时加载数据。
    __getitem__ 返回行索引；collate_fn 在 batch 级别从 HDF5 读取连续切片并转换为 tensor。
    需配合 DataLoader(shuffle=False) 使用以保证批次连续，提高读取效率。
    """
    def __init__(self, split_handle: SplitHandle, start_offset: int = 0, length: Optional[int] = None, cache_blocks: int = 8):
        self.handle = split_handle
        self.start_offset = int(start_offset)
        base_total = len(split_handle)
        self.total = int(base_total - self.start_offset) if length is None else int(min(length, base_total - self.start_offset))
        self.label_col = split_handle.label_col
        self._cache_blocks = max(0, int(cache_blocks))
        self._lru: "OrderedDict[Tuple[int,int], pd.DataFrame]" = OrderedDict()
        # Lazy, per-process read-only HDFStore (not pickled)
        self._store: Optional[pd.HDFStore] = None

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int):
        return int(self.start_offset + idx)

    # Ensure a read-only HDFStore, one per process
    def _get_store(self) -> pd.HDFStore:
        store = getattr(self, "_store", None)
        if store is None or not store.is_open:
            # mode='r' avoids unnecessary write locks
            self._store = pd.HDFStore(self.handle.path, mode='r')
        return self._store

    def __getstate__(self):
        state = self.__dict__.copy()
        # Do not pickle open HDFStore; it is not fork-safe
        state['_store'] = None
        return state

    def __del__(self):
        try:
            store = getattr(self, "_store", None)
            if store is not None:
                store.close()
        except Exception:
            pass

    def _read_range(self, start: int, stop: int) -> pd.DataFrame:
        key = (int(start), int(stop))
        if self._cache_blocks > 0 and key in self._lru:
            df = self._lru.pop(key)
            self._lru[key] = df
            return df
        store = self._get_store()
        # Use storer.select for efficient range reads without reopening
        storer = store.get_storer('data')
        df = storer.read(start=start, stop=stop)
        # Ensure the DataFrame carries absolute row indices [start, stop),
        # because some HDF5 stores are written without index and pandas will
        # default to a 0-based RangeIndex upon slicing. Downstream callers may
        # rely on df.loc[absolute_indices].
        df.index = pd.RangeIndex(start, stop)
        if self._cache_blocks > 0:
            self._lru[key] = df
            if len(self._lru) > self._cache_blocks:
                self._lru.popitem(last=False)
        return df

    def collate_fn(self, batch_indices: List[int]):
        if not batch_indices:
            empty_num = torch.empty((0, len(self.handle.num_cols)), dtype=torch.float32) if self.handle.num_cols else torch.empty((0, 0), dtype=torch.float32)
            empty_cat = torch.empty((0, len(self.handle.cat_cols)), dtype=torch.long)
            empty_lbl = torch.empty((0, 1), dtype=torch.float32)
            return empty_num, empty_cat, empty_lbl

        sorted_idx = sorted(int(i) for i in batch_indices)
        ranges: List[Tuple[int, int]] = []
        s = sorted_idx[0]
        prev = s
        for i in sorted_idx[1:]:
            if i == prev + 1:
                prev = i
            else:
                ranges.append((s, prev + 1))
                s = i
                prev = i
        ranges.append((s, prev + 1))

        dfs = [self._read_range(st, sp) for st, sp in ranges]
        df_batch = pd.concat(dfs, axis=0, ignore_index=True)

        if self.handle.num_cols:
            numerical_values = df_batch[self.handle.num_cols].values
            numerical_data = torch.tensor(numerical_values, dtype=torch.float32)
        else:
            numerical_data = torch.empty((len(df_batch), 0), dtype=torch.float32)
        categorical_values = df_batch[self.handle.cat_cols].values
        categorical_data = torch.tensor(categorical_values, dtype=torch.long)
        labels = torch.tensor(df_batch[self.label_col].values, dtype=torch.float32).unsqueeze(1)
        return numerical_data, categorical_data, labels


class StreamingReplayDataset:
    def __init__(self, handle: SplitHandle, numerical_cols: List[str], categorical_cols: List[str], dataset_name: str, replay_batch_size: int):
        self.handle = handle
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.dataset_name = dataset_name
        self.replay_batch_size = replay_batch_size
        self.total_samples = len(handle)
        self.label_col = 'label' if dataset_name in ['criteo', 'criteo_ctr_full'] else 'click'

    def _get_store(self) -> pd.HDFStore:
        store = getattr(self, "_store", None)
        if store is None or not store.is_open:
            self._store = pd.HDFStore(self.handle.path, mode='r')
        return self._store

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_store'] = None
        return state

    def __del__(self):
        try:
            store = getattr(self, "_store", None)
            if store is not None:
                store.close()
        except Exception:
            pass

    def _read_indices(self, indices: np.ndarray) -> pd.DataFrame:
        if indices.size == 0:
            return pd.DataFrame(columns=[self.label_col] + self.numerical_cols + self.categorical_cols)
        idx_sorted = np.sort(indices)
        ranges: List[Tuple[int, int]] = []
        s = int(idx_sorted[0])
        prev = s
        for i in idx_sorted[1:]:
            i = int(i)
            if i == prev + 1:
                prev = i
            else:
                ranges.append((s, prev + 1))
                s = i
                prev = i
        ranges.append((s, prev + 1))
        store = self._get_store()
        storer = store.get_storer('data')
        dfs = []
        for st, sp in ranges:
            df_part = storer.read(start=st, stop=sp)
            # Align index to absolute positions so df.loc[indices] succeeds
            df_part.index = pd.RangeIndex(st, sp)
            dfs.append(df_part)
        df = pd.concat(dfs, axis=0, ignore_index=False)
        return df.loc[indices]

    def sample_batch(self):
        sampled_indices = np.random.choice(self.total_samples, size=self.replay_batch_size, replace=True)
        df = self._read_indices(sampled_indices)
        if self.numerical_cols:
            x_num = torch.tensor(df[self.numerical_cols].values, dtype=torch.float32)
        else:
            x_num = torch.empty((len(df), 0), dtype=torch.float32)
        x_cat = torch.tensor(df[self.categorical_cols].values, dtype=torch.long)
        y = torch.tensor(df[self.label_col].values, dtype=torch.float32).unsqueeze(1)
        return x_num, x_cat, y


class StreamingRowDataset(Dataset):
    def __init__(self, split_handle: SplitHandle, page_size: int = 8192):
        self.handle = split_handle
        self.total = len(split_handle)
        self.page_size = max(1024, int(page_size))
        self.label_col = split_handle.label_col
        self._page_start: Optional[int] = None
        self._page_df: Optional[pd.DataFrame] = None
        self._store: Optional[pd.HDFStore] = None

    def __len__(self):
        return self.total

    def _load_page(self, page_start: int):
        page_stop = min(page_start + self.page_size, self.total)
        store = self._get_store()
        storer = store.get_storer('data')
        df = storer.read(start=page_start, stop=page_stop)
        self._page_start = page_start
        self._page_df = df

    def _get_store(self) -> pd.HDFStore:
        store = getattr(self, "_store", None)
        if store is None or not store.is_open:
            self._store = pd.HDFStore(self.handle.path, mode='r')
        return self._store

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_store'] = None
        return state

    def __del__(self):
        try:
            store = getattr(self, "_store", None)
            if store is not None:
                store.close()
        except Exception:
            pass

    def __getitem__(self, idx: int):
        idx = int(idx)
        if self._page_start is None or self._page_df is None or not (self._page_start <= idx < self._page_start + len(self._page_df)):
            page_start = (idx // self.page_size) * self.page_size
            self._load_page(page_start)
        off = idx - self._page_start
        row = self._page_df.iloc[off:off+1]
        if self.handle.num_cols:
            x_num = torch.tensor(row[self.handle.num_cols].values, dtype=torch.float32)
        else:
            x_num = torch.empty((1, 0), dtype=torch.float32)
        x_cat = torch.tensor(row[self.handle.cat_cols].values, dtype=torch.long)
        y = torch.tensor(row[self.label_col].values, dtype=torch.float32).unsqueeze(1)
        return x_num.squeeze(0), x_cat.squeeze(0), y.squeeze(0)


class StreamingIndexManager:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

    def _paths(self, split_name: str):
        return {
            'labels': os.path.join(self.index_dir, f"{split_name}_labels.int8"),
            'pos_idx': os.path.join(self.index_dir, f"{split_name}_pos_idx.npy"),
            'meta': os.path.join(self.index_dir, f"{split_name}_meta.json"),
        }

    def ensure_split_index(self, split_name: str, handle: SplitHandle):
        paths = self._paths(split_name)
        up_to_date = False
        if os.path.exists(paths['meta']):
            try:
                with open(paths['meta'], 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                if meta.get('nrows') == len(handle) and meta.get('src_path') == handle.path:
                    up_to_date = True
            except Exception:
                up_to_date = False
        if up_to_date and os.path.exists(paths['labels']) and os.path.exists(paths['pos_idx']):
            return paths
        N = len(handle)
        labels_mm = np.memmap(paths['labels'], mode='w+', dtype=np.int8, shape=(N,))
        pos_list: List[int] = []
        step = 1_000_000
        for start in range(0, N, step):
            stop = min(start + step, N)
            df = pd.read_hdf(handle.path, key='data', start=start, stop=stop)
            y = df[handle.label_col].astype(np.int8).values
            labels_mm[start:stop] = y
            if (y == 1).any():
                rel = np.nonzero(y == 1)[0]
                pos_list.append(rel + start)
            del df
        del labels_mm
        if pos_list:
            pos_idx = np.concatenate(pos_list).astype(np.int64)
        else:
            pos_idx = np.empty((0,), dtype=np.int64)
        np.save(paths['pos_idx'], pos_idx)
        with open(paths['meta'], 'w', encoding='utf-8') as f:
            json.dump({'nrows': N, 'src_path': handle.path, 'ts': time.time()}, f)
        return paths

    def load_labels_memmap(self, labels_path: str) -> np.memmap:
        return np.memmap(labels_path, mode='r', dtype=np.int8)


class _DummyIndexDataset(Dataset):
    def __init__(self, length: int):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, idx: int):
        return idx


class StreamingSamplerCollate:
    def __init__(self, handle: SplitHandle, index_mgr: StreamingIndexManager, split_name: str,
                 batch_size: int, strategy: str, target_pos_rate: float = 0.5, seed: int = 42):
        self.handle = handle
        self.batch_size = int(batch_size)
        self.strategy = strategy
        self.target_pos_rate = float(target_pos_rate)
        self.rng = np.random.default_rng(seed)
        self.idx_paths = index_mgr.ensure_split_index(split_name, handle)
        self.labels_mm = index_mgr.load_labels_memmap(self.idx_paths['labels'])
        self.pos_idx = np.load(self.idx_paths['pos_idx']) if os.path.exists(self.idx_paths['pos_idx']) else np.empty((0,), dtype=np.int64)
        self.reader = StreamingHDF5IndexDataset(handle, cache_blocks=8)
        self.N = len(handle)

    def _sample_temporal(self, k: int) -> np.ndarray:
        N = self.N
        T = N * (N + 1) // 2
        r = self.rng.integers(1, T + 1, size=k, endpoint=True)
        i = ((np.sqrt(1 + 8 * r) - 1) // 2).astype(np.int64) - 1
        i = np.clip(i, 0, N - 1)
        return i

    def _sample_biased(self, k: int) -> np.ndarray:
        num_pos = int(round(k * self.target_pos_rate))
        num_neg = k - num_pos
        out = []
        if self.pos_idx.size > 0 and num_pos > 0:
            sel = self.rng.integers(0, len(self.pos_idx), size=num_pos)
            out.append(self.pos_idx[sel])
        if num_neg > 0:
            needed = num_neg
            buf = []
            while needed > 0:
                cand = self.rng.integers(0, self.N, size=needed)
                mask = (self.labels_mm[cand] == 0)
                if mask.any():
                    picks = cand[mask]
                    buf.append(picks)
                    needed -= len(picks)
            if buf:
                out.append(np.concatenate(buf))
        if out:
            return np.concatenate(out)
        return self.rng.integers(0, self.N, size=k)

    def _sample_combined(self, k: int) -> np.ndarray:
        num_pos = int(round(k * self.target_pos_rate))
        num_neg = k - num_pos
        out = []
        if self.pos_idx.size > 0 and num_pos > 0:
            Np = len(self.pos_idx)
            T = Np * (Np + 1) // 2
            r = self.rng.integers(1, T + 1, size=num_pos, endpoint=True)
            j = ((np.sqrt(1 + 8 * r) - 1) // 2).astype(np.int64) - 1
            j = np.clip(j, 0, Np - 1)
            out.append(self.pos_idx[j])
        if num_neg > 0:
            needed = num_neg
            buf = []
            while needed > 0:
                cand = self._sample_temporal(needed)
                mask = (self.labels_mm[cand] == 0)
                if mask.any():
                    picks = cand[mask]
                    buf.append(picks)
                    needed -= len(picks)
            if buf:
                out.append(np.concatenate(buf))
        if out:
            return np.concatenate(out)
        return self._sample_temporal(k)

    def __call__(self, _batch_indices: List[int]):
        k = self.batch_size
        if self.strategy == 'random':
            idx = self.rng.integers(0, self.N, size=k)
        elif self.strategy == 'temporal':
            idx = self._sample_temporal(k)
        elif self.strategy == 'biased_sample':
            idx = self._sample_biased(k)
        elif self.strategy == 'combined':
            idx = self._sample_combined(k)
        else:
            idx = self.rng.integers(0, self.N, size=k)

        arr = np.sort(idx).tolist()
        ranges: List[Tuple[int, int]] = []
        if arr:
            s = arr[0]
            prev = s
            for v in arr[1:]:
                if v == prev + 1:
                    prev = v
                else:
                    ranges.append((s, prev + 1))
                    s = v
                    prev = v
            ranges.append((s, prev + 1))
        if ranges:
            dfs = [self.reader._read_range(st, sp) for st, sp in ranges]
            df = pd.concat(dfs, axis=0, ignore_index=False)
            df = df.loc[idx]
        else:
            df = pd.DataFrame(columns=[self.handle.label_col] + self.handle.num_cols + self.handle.cat_cols)
        if self.handle.num_cols:
            x_num = torch.tensor(df[self.handle.num_cols].values, dtype=torch.float32)
        else:
            x_num = torch.empty((len(df), 0), dtype=torch.float32)
        x_cat = torch.tensor(df[self.handle.cat_cols].values, dtype=torch.long)
        y = torch.tensor(df[self.handle.label_col].values, dtype=torch.float32).unsqueeze(1)
        return x_num, x_cat, y
