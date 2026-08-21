import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple

import os
import glob
import math
import time
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
class NodeEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_linear_regression_slope(time_vec: torch.Tensor,
                                   y: torch.Tensor,
                                   mask: torch.Tensor) -> torch.Tensor:
    """
    对每个样本做 1D 线性回归 y(t)=a*t+b，返回 slope a
    time_vec: [K]
    y:        [nM, K]
    mask:     [nM, K]  (0/1)
    return:   [nM]
    """
    msum = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    t_mean = (time_vec.unsqueeze(0) * mask).sum(1, keepdim=True) / msum
    y_mean = (y * mask).sum(1, keepdim=True) / msum
    t_c = (time_vec.unsqueeze(0) - t_mean) * mask
    y_c = (y - y_mean) * mask
    num = (t_c * y_c).sum(dim=1)
    den = (t_c * t_c).sum(dim=1).clamp_min(1e-8)
    return num / den


class EdgeScorer(nn.Module):
    """
    concat(hi, hj, dx, dy, tau_norm) -> logit
    其中 dx,dy = xj - (xi + τ * v_hat)
    """
    def __init__(self, node_dim: int, hidden: int = 128, extra_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim * 2 + extra_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, hi: torch.Tensor, hj: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        nM, H = hi.shape
        nT = hj.shape[0]
        hi_rep = hi[:, None, :].expand(nM, nT, H)
        hj_rep = hj[None, :, :].expand(nM, nT, H)
        feat = torch.cat([hi_rep, hj_rep, extra], dim=-1)    # [nM,nT,2H+E]
        return self.net(feat).squeeze(-1)                    # [nM,nT]


class AssocModel(nn.Module):
    """
    历史K帧 -> 速度回归 (x,y) -> 当前位置 + τ*速度 -> 与未来候选计算边打分 + dummy 列
    """
    def __init__(self, node_in_dim: int, node_hidden: int = 128,
                 node_out: int = 128, edge_hidden: int = 128):
        super().__init__()
        self.enc_curr = NodeEncoder(node_in_dim, node_hidden, node_out)
        self.enc_fut  = NodeEncoder(node_in_dim, node_hidden, node_out)
        self.edge     = EdgeScorer(node_out, hidden=edge_hidden, extra_dim=3)
        self.dummy_bias = nn.Parameter(torch.tensor(0.0))

    def estimate_velocity_xy(self, hist_xy: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        """
        hist_xy:   [nM, K, 2]   （按ID对齐的历史坐标，最后一列是当前帧）
        hist_mask: [nM, K]      （0/1）
        return:    [nM, 2]      v_hat
        """
        nM, K, _ = hist_xy.shape
        device = hist_xy.device
        t = torch.arange(-(K-1), 1, device=device, dtype=hist_xy.dtype)  # [-(K-1)..0]
        vx = masked_linear_regression_slope(t, hist_xy[:, :, 0], hist_mask)
        vy = masked_linear_regression_slope(t, hist_xy[:, :, 1], hist_mask)
        return torch.stack([vx, vy], dim=-1)

    def forward_one_tau(self,
                        curr_nodes: torch.Tensor,   # [nM,4] -> [x,y,w,h]
                        hist_xy: torch.Tensor,      # [nM,K,2]
                        hist_mask: torch.Tensor,    # [nM,K]
                        fut_nodes: torch.Tensor,    # [nT,4]
                        tau: int, Nmax: int) -> torch.Tensor:
        device = curr_nodes.device
        nM = curr_nodes.shape[0]
        nT = fut_nodes.shape[0]

        hi = self.enc_curr(curr_nodes)   # [nM,H]
        hj = self.enc_fut(fut_nodes)     # [nT,H]

        xi = curr_nodes[:, 0:2]          # [nM,2]
        v_hat = self.estimate_velocity_xy(hist_xy, hist_mask)  # [nM,2]

        tau_f = torch.tensor(float(tau), device=device)
        x_pred = xi + tau_f * v_hat                         # [nM,2]
        xj = fut_nodes[:, 0:2] if nT > 0 else torch.zeros(0, 2, device=device)

        if nT > 0:
            dx = xj[None, :, 0] - x_pred[:, None, 0]       # [nM,nT]
            dy = xj[None, :, 1] - x_pred[:, None, 1]       # [nM,nT]
            tau_norm = (tau_f / float(max(Nmax, 1))) * torch.ones_like(dx)
            extra = torch.stack([dx, dy, tau_norm], dim=-1)    # [nM,nT,3]
            logits_nt = self.edge(hi, hj, extra)               # [nM,nT]
        else:
            logits_nt = torch.empty(nM, 0, device=device)

        dummy_col = self.dummy_bias * torch.ones(nM, 1, device=device, dtype=logits_nt.dtype)
        logits = torch.cat([logits_nt, dummy_col], dim=1)       # [nM, nT+1]
        return logits

    def forward(self,
                curr_nodes: torch.Tensor,
                hist_xy: torch.Tensor,
                hist_mask: torch.Tensor,
                fut_nodes_list: List[torch.Tensor]) -> List[torch.Tensor]:
        N = len(fut_nodes_list)
        return [self.forward_one_tau(curr_nodes, hist_xy, hist_mask, fut_nodes_list[tau-1], tau, N)
                for tau in range(1, N+1)]
        

class MotSequenceWindowDataset(Dataset):
    """
    直接读取序列级 NPZ（frames, dets, ids），在 __init__ 时枚举所有滑窗位置 (file, M)。

    每次 __getitem__ 返回一个样本（字典）：
      K, N, curr_nodes [nM,4], hist_xy [nM,K,2], hist_mask [nM,K],
      fut_nodes_list: list of [n_t,4],
      targets_list:   list of [nM] (值域 0..n_t，其中 n_t 表示 dummy)
    """
    def __init__(self, npz_files: List[str], K: int, N: int, stride: int = 1, min_curr: int = 1):
        self.files = sorted(npz_files)
        self.K = K
        self.N = N
        self.stride = max(1, int(stride))
        self.min_curr = min_curr

        self.index: List[Tuple[int, int]] = []  # (file_idx, M)
        self.meta: List[Dict[str, Any]] = []
        for fi, path in enumerate(self.files):
            z = np.load(path, allow_pickle=True)
            frames = int(np.array(z["frames"]).item())
            dets = z["dets"]
            ids  = z["ids"]
            z.close()

            for M in range(self.K - 1, frames - self.N - 1 + 1, self.stride):
                detM = dets[M]
                if detM is None or detM.dtype == object:
                    detM = detM.tolist()
                nM = detM.shape[0] if isinstance(detM, np.ndarray) else len(detM)
                if nM >= self.min_curr:
                    self.index.append((fi, M))
            self.meta.append({"frames": frames})

        self._cache_path = None
        self._cache_data = None

    def __len__(self):
        return len(self.index)

    def _load_file(self, fi: int):
        path = self.files[fi]
        if self._cache_path != path:
            z = np.load(path, allow_pickle=True)
            dets = [np.array(a, dtype=np.float32) for a in z["dets"].tolist()]
            ids  = [np.array(a, dtype=np.int64)  for a in z["ids"].tolist()]
            frames = int(np.array(z["frames"]).item())
            z.close()
            self._cache_path = path
            self._cache_data = {"dets": dets, "ids": ids, "frames": frames}
        return self._cache_data

    def __getitem__(self, idx):
        fi, M = self.index[idx]
        data = self._load_file(fi)
        dets = data["dets"]
        ids  = data["ids"]

        curr_nodes = dets[M]                       # [nM,4]
        ids_M = ids[M]                             # [nM]
        nM = curr_nodes.shape[0]

        # ---------- 历史对齐 (基于ID) ----------
        K = self.K
        hist_xy  = np.zeros((nM, K, 2), dtype=np.float32)
        hist_msk = np.zeros((nM, K), dtype=np.float32)
        t_list = list(range(M - K + 1, M + 1))
        for i in range(nM):
            tid = int(ids_M[i])
            for k, t in enumerate(t_list):
                if t < 0:
                    continue
                ids_t = ids[t]
                det_t = dets[t]
                if ids_t.shape[0] == 0:
                    continue
                idxs = np.where(ids_t == tid)[0]
                if idxs.size > 0:
                    j = int(idxs[0])
                    hist_xy[i, k, :] = det_t[j, 0:2]  # 只取 (x,y)
                    hist_msk[i, k] = 1.0
        hist_xy[:, -1, :] = curr_nodes[:, 0:2]
        hist_msk[:, -1] = 1.0

        # ---------- 未来帧 & 真值指派 ----------
        N = self.N
        fut_nodes_list = []
        targets_list = []
        for tau in range(1, N + 1):
            t = M + tau
            fut = dets[t]                        # [n_t,4]
            fut_ids = ids[t]                     # [n_t]
            n_t = fut.shape[0]
            targets = np.full((nM,), fill_value=n_t, dtype=np.int64)
            if n_t > 0:
                id2idx = {}
                for j, tj in enumerate(fut_ids.tolist()):
                    if tj not in id2idx:
                        id2idx[tj] = j
                for i in range(nM):
                    tid = int(ids_M[i])
                    if tid in id2idx:
                        targets[i] = id2idx[tid]
            fut_nodes_list.append(fut.astype(np.float32))
            targets_list.append(targets)

        sample = {
            "K": K, "N": N,
            "curr_nodes": curr_nodes.astype(np.float32),     # [nM,4]
            "hist_xy": hist_xy.astype(np.float32),           # [nM,K,2]
            "hist_mask": hist_msk.astype(np.float32),        # [nM,K]
            "fut_nodes_list": fut_nodes_list,                # list of [n_t,4]
            "targets_list": targets_list,                    # list of [nM]
            "weights": None,
            "meta": {"file": self.files[fi], "M": M},
        }
        return sample


def collate_keep_list(batch: List[Dict[str, Any]]):
    return batch



def infer(model, loader, device):
    model.eval()
    accuracy_per_tau = {tau: [] for tau in range(1, loader.dataset.K+1)}  # 用于存储每个τ的准确率
    result_data = {}

    for batch in loader:
        for s in batch:
            curr = torch.from_numpy(s["curr_nodes"]).to(device)
            hist_xy = torch.from_numpy(s["hist_xy"]).to(device)
            hist_mask = torch.from_numpy(s["hist_mask"]).to(device)
            fut_list = [torch.from_numpy(x).to(device) for x in s["fut_nodes_list"]]
            tgt_list = [torch.from_numpy(x).to(device) for x in s["targets_list"]]

            logits_list = model(curr, hist_xy, hist_mask, fut_list)

            for tau_idx, (logits, tgt) in enumerate(zip(logits_list, tgt_list), start=1):
                pred = logits.argmax(dim=1)  # 获取预测值
                correct = (pred == tgt).sum().item()
                accuracy_per_tau[tau_idx].append(correct / float(len(tgt)))  # 计算并保存准确率

                # 存储当前帧与未来帧的关联结果
                key = f"{s['meta']['file']}_{s['meta']['M']}_{tau_idx}"  # 将元组 (file, M, tau) 转换为字符串
                result_data[key] = {
                    "correct": correct,
                    "total": len(tgt),
                    "accuracy": correct / float(len(tgt)),
                }

    avg_accuracy_per_tau = {tau: np.mean(acc) for tau, acc in accuracy_per_tau.items()}

    return result_data, avg_accuracy_per_tau


import argparse
import glob
import torch
import json
import os
from torch.utils.data import DataLoader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    ap.add_argument("--data_dir", type=str, required=True, help="包含 *.npz 的测试数据目录")
    ap.add_argument("--K", type=int, default=5, help="历史帧数和预测的最大帧数")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--save_dir", type=str, default="./outputs_infer", help="推理结果保存路径")
    ap.add_argument("--output_json", type=str, default="infer_results.json", help="推理结果 JSON 保存路径")
    ap.add_argument("--N", type=int, default=3, help="预测未来帧数")
    ap.add_argument("--stride", type=int, default=1, help="滑窗步长")
    ap.add_argument("--min_curr", type=int, default=1, help="当前帧最少检测数")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    # 加载模型
    model = AssocModel(node_in_dim=4, node_hidden=128, node_out=128, edge_hidden=128).to(device)
    model.load_state_dict(torch.load(args.model_path)["model_state"])
    model.eval()

    # 准备数据集
    files = glob.glob(os.path.join(args.data_dir, "*.npz"))
    assert len(files) > 0, f"未在 {args.data_dir} 找到 .npz"
    
    dataset = MotSequenceWindowDataset(files, K=args.K, N=args.N,
                                       stride=args.stride, min_curr=args.min_curr)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_keep_list)
    # 推理
    result_data, avg_accuracy_per_tau = infer(model, loader, device)

    # 保存每对关联结果到 JSON 文件
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)

    # 打印每个 τ 的平均准确率
    print(f"每对帧的关联准确率：")
    for tau, acc in avg_accuracy_per_tau.items():
        print(f"τ={tau}: {acc:.4f}")

    print(f"推理完成，结果保存在 {args.save_dir} 和 {args.output_json}")


if __name__ == "__main__":
    main()


#python graph_assoc_M_infer.py --model_path "D:\CMH\MyMotionPredictor\outputs_from_mot_npz\best_model.pt" --data_dir "D:\CMH\MyMotionPredictor\NPZtest" --save_dir "D:\CMH\MyMotionPredictor\outputs_infer" --output_json "D:\CMH\MyMotionPredictor\infer_results.json" --K 5