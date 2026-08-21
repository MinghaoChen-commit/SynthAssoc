# -*- coding: utf-8 -*-
"""
Train temporal association directly from MOT-style NPZ sequences.

输入 NPZ（由 preprocess_mot_csv_to_npz.py 生成）字段：
- frames: int
- dets  : object数组，len==frames，每帧 [Ni,4] float32，列为 [x,y,w,h]
- ids   : object数组，len==frames，每帧 [Ni]   int64，对应 GT track id

本脚本：
- 构造滑窗样本：(K历史 + 当前M) -> 预测 M 与 M+1..M+N 的关联
- 历史速度估计：按ID对齐的 K 帧历史坐标做线性回归
- 损失：各 τ 行向交叉熵的加权和（两帧关联加权求和）
- 验证阶段打印“按 τ 聚合”的准确率（M→M+1, M→M+2, ...）
"""

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


# ============== 工具 ==============

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def exp_decay_weights(N: int, gamma: float = 0.4) -> torch.Tensor:
    w = torch.tensor([math.exp(-gamma * t) for t in range(N)], dtype=torch.float32)
    return w / (w.sum() + 1e-8)


# ============== 模型 ==============

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


# ============== 数据集 ==============

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


# ============== 损失与评估 ==============

def weighted_assoc_loss(logits_per_tau: List[torch.Tensor],
                        targets_list: List[torch.Tensor],
                        weights: torch.Tensor) -> torch.Tensor:
    losses = []
    for w, logits, tgt in zip(weights, logits_per_tau, targets_list):
        losses.append(w * F.cross_entropy(logits, tgt))
    return torch.stack(losses).sum()


@torch.no_grad()
def eval_batch(logits_per_tau: List[torch.Tensor],
               targets_list: List[torch.Tensor],
               weights: torch.Tensor) -> Dict[str, float]:
    accs = []
    for logits, tgt in zip(logits_per_tau, targets_list):
        pred = logits.argmax(dim=1)
        accs.append((pred == tgt).float().mean().item())
    accs = np.array(accs, dtype=np.float32)
    w = weights.detach().cpu().numpy()
    return {"acc_mean": float(accs.mean()), "acc_w": float((accs * w).sum())}


@torch.no_grad()
def accumulate_per_tau_stats(logits_per_tau, targets_list):
    """
    返回两个列表（长度 N）：
      - correct_per_tau: 每个 τ 上预测命中的个数（逐窗口逐行累加）
      - total_per_tau:   每个 τ 上的总行数
    """
    correct_per_tau, total_per_tau = [], []
    for logits, tgt in zip(logits_per_tau, targets_list):
        pred = logits.argmax(dim=1)          # [nM]
        correct = (pred == tgt).sum().item()
        total = tgt.numel()
        correct_per_tau.append(correct)
        total_per_tau.append(total)
    return correct_per_tau, total_per_tau


# ============== 训练/验证 ==============

def train_one_epoch(model, loader, optimizer, device, epoch, gamma=0.4, log_interval=50):
    model.train()
    total_loss, total_acc_w, n_samples = 0.0, 0.0, 0
    for step, batch in enumerate(loader):
        for s in batch:
            n_samples += 1
            curr = torch.from_numpy(s["curr_nodes"]).to(device)                # [nM,4]
            hist_xy = torch.from_numpy(s["hist_xy"]).to(device)               # [nM,K,2]
            hist_mask = torch.from_numpy(s["hist_mask"]).to(device)           # [nM,K]
            fut_list = [torch.from_numpy(x).to(device) for x in s["fut_nodes_list"]]
            tgt_list = [torch.from_numpy(x).to(device) for x in s["targets_list"]]
            N = s["N"]
            weights = exp_decay_weights(N, gamma=gamma).to(device)

            optimizer.zero_grad()
            logits_list = model(curr, hist_xy, hist_mask, fut_list)
            loss = weighted_assoc_loss(logits_list, tgt_list, weights)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metrics = eval_batch(logits_list, tgt_list, weights)
            total_loss += loss.item()
            total_acc_w += metrics["acc_w"]

        if (step + 1) % log_interval == 0:
            print(f"[Train] epoch {epoch} step {step+1}/{len(loader)} "
                  f"loss={total_loss/max(n_samples,1):.4f} acc_w={total_acc_w/max(n_samples,1):.4f}")
    return {"loss": total_loss/max(n_samples,1), "acc_w": total_acc_w/max(n_samples,1)}


@torch.no_grad()
def validate(model, loader, device, gamma=0.4):
    model.eval()
    total_loss, total_acc_w, n_samples = 0.0, 0.0, 0

    per_tau_correct_sum = None
    per_tau_total_sum = None
    N_fixed = None

    for batch in loader:
        for s in batch:
            n_samples += 1
            curr = torch.from_numpy(s["curr_nodes"]).to(device)
            hist_xy = torch.from_numpy(s["hist_xy"]).to(device)
            hist_mask = torch.from_numpy(s["hist_mask"]).to(device)
            fut_list = [torch.from_numpy(x).to(device) for x in s["fut_nodes_list"]]
            tgt_list = [torch.from_numpy(x).to(device) for x in s["targets_list"]]
            N = s["N"]
            if N_fixed is None:
                N_fixed = N
                per_tau_correct_sum = np.zeros(N_fixed, dtype=np.int64)
                per_tau_total_sum   = np.zeros(N_fixed, dtype=np.int64)

            weights = exp_decay_weights(N, gamma=gamma).to(device)

            logits_list = model(curr, hist_xy, hist_mask, fut_list)
            loss = weighted_assoc_loss(logits_list, tgt_list, weights)
            metrics = eval_batch(logits_list, tgt_list, weights)

            total_loss += loss.item()
            total_acc_w += metrics["acc_w"]

            # 逐 τ 累加命中/总数
            c, t = accumulate_per_tau_stats(logits_list, tgt_list)
            per_tau_correct_sum += np.array(c, dtype=np.int64)
            per_tau_total_sum   += np.array(t, dtype=np.int64)

    # 计算并打印 val 上“当前帧→未来 M+τ”的准确率（逐 τ 汇总）
    per_tau_acc = (per_tau_correct_sum / np.maximum(per_tau_total_sum, 1)).astype(np.float32)
    tau_summary = " | ".join([f"τ={i+1}: acc={per_tau_acc[i]:.4f}" for i in range(N_fixed)])
    print(f"[Val per-τ] {tau_summary}")

    return {
        "loss": total_loss/max(n_samples,1),
        "acc_w": total_acc_w/max(n_samples,1),
        "per_tau_acc": per_tau_acc.tolist(),
    }


# ============== 主程序 ==============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="包含 *.npz 的目录")
    ap.add_argument("--K", type=int, default=4, help="历史帧数")
    ap.add_argument("--N", type=int, default=3, help="预测未来帧数")
    ap.add_argument("--stride", type=int, default=1, help="滑窗步长")
    ap.add_argument("--min_curr", type=int, default=1, help="当前帧最少检测数")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1, help="建议=1（变长样本）")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--split", type=float, default=0.85, help="训练集比例")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.4, help="时间权重指数衰减")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="./outputs_from_mot_npz")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    files = glob.glob(os.path.join(args.data_dir, "*.npz"))
    assert len(files) > 0, f"未在 {args.data_dir} 找到 .npz"

    node_in_dim = 4  # [x,y,w,h]
    print(f"Node feature dim = {node_in_dim} (x,y,w,h)")

    dataset = MotSequenceWindowDataset(files, K=args.K, N=args.N,
                                       stride=args.stride, min_curr=args.min_curr)
    n_total = len(dataset)
    assert n_total > 0, "没有可用样本窗口，请检查 K/N/stride 与数据覆盖。"
    n_train = max(1, int(n_total * args.split))
    n_val = max(1, n_total - n_train)
    if n_train + n_val > n_total:
        n_val = n_total - n_train

    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_keep_list)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_keep_list)

    model = AssocModel(node_in_dim=node_in_dim,
                       node_hidden=args.hidden,
                       node_out=args.hidden,
                       edge_hidden=args.hidden).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_path = float("inf"), os.path.join(args.save_dir, "best_model.pt")
    log_path = os.path.join(args.save_dir, "train_log.jsonl")
    t0 = time.time()

    with open(log_path, "w", encoding="utf-8") as flog:
        for ep in range(1, args.epochs + 1):
            tr = train_one_epoch(model, train_loader, optimizer, device, ep, gamma=args.gamma)
            va = validate(model, val_loader, device, gamma=args.gamma)

            rec = {"epoch": ep, "train_loss": tr["loss"], "train_acc_w": tr["acc_w"],
                   "val_loss": va["loss"], "val_acc_w": va["acc_w"],
                   "per_tau_acc": va["per_tau_acc"]}
            flog.write(json.dumps(rec, ensure_ascii=False) + "\n"); flog.flush()

            print(f"[Epoch {ep}] "
                  f"train_loss={tr['loss']:.4f} acc_w={tr['acc_w']:.4f} | "
                  f"val_loss={va['loss']:.4f} acc_w={va['acc_w']:.4f} | "
                  f"per-τ acc={['{:.4f}'.format(x) for x in va['per_tau_acc']]}")

            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save({
                    "model_state": model.state_dict(),
                    "meta": {"node_in_dim": node_in_dim, "K": args.K, "N": args.N},
                    "args": vars(args),
                }, best_path)
                print(f"  -> 保存最优模型到: {best_path}")

    print(f"训练完成 | 最优验证loss={best_val:.4f} | 用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()

# 2) 训练：利用历史 K=4 帧预测未来 N=3 帧的关联（用 GPU）
#python graph_assoc_M_train.py --data_dir "D:\CMH\MyMotionPredictor\NPZ" --K 5 --N 3 --stride 1 --epochs 30 --batch_size 1 --hidden 128 --lr 1e-3 --gamma 0.4 --device cuda --save_dir "D:\CMH\MyMotionPredictor\outputs_from_mot_npz"

#python graph_assoc_M_train.py --data_dir "D:\CMH\MyMotionPredictor\NPZtrainbiod" --K 5 --N 3 --stride 1 --epochs 30 --batch_size 1 --hidden 128 --lr 1e-3 --gamma 0.4 --device cuda --save_dir "D:\CMH\MyMotionPredictor\outputs_from_mot_npz_biod"

#python graph_assoc_M_train.py --data_dir "D:\CMH\MyMotionPredictor\NPZtrainSoccernet" --K 5 --N 3 --stride 1 --epochs 30 --batch_size 1 --hidden 128 --lr 1e-3 --gamma 0.4 --device cuda --save_dir "D:\CMH\MyMotionPredictor\outputs_from_mot_npz_soccernet"

#python graph_assoc_M_train.py --data_dir "D:\CMH\MyMotionPredictor\NPZtrainGMOT" --K 5 --N 3 --stride 1 --epochs 30 --batch_size 1 --hidden 128 --lr 1e-3 --gamma 0.4 --device cuda --save_dir "D:\CMH\MyMotionPredictor\outputs_from_mot_npz_gmot\3"

#python graph_assoc_M_train.py --data_dir "D:\CMH\MyMotionPredictor\NPZtrainDanceTrack" --K 5 --N 3 --stride 1 --epochs 30 --batch_size 1 --hidden 128 --lr 1e-3 --gamma 0.4 --device cuda --save_dir "D:\CMH\MyMotionPredictor\outputs_from_mot_npz_danceterack\3"
