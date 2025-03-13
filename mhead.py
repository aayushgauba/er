#!/usr/bin/env python
# mhead.py

import os
import sys
sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

from omegaconf import OmegaConf
from torch.optim.lr_scheduler import StepLR

# local references 
from utils.data_utils import SE3Demo, seg_pointcloud, random_dropout, mask_part_point_cloud
from utils.loss_utils import double_geodesic_distance_between_poses

###############################################################################
# Quantum MHA with Key-Transpose Fix
###############################################################################
class QuantumMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=16, num_heads=2):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q/K/V real+imag
        self.qr = nn.Linear(embed_dim, embed_dim)
        self.qi = nn.Linear(embed_dim, embed_dim)
        self.kr = nn.Linear(embed_dim, embed_dim)
        self.ki = nn.Linear(embed_dim, embed_dim)
        self.vr = nn.Linear(embed_dim, embed_dim)
        self.vi = nn.Linear(embed_dim, embed_dim)

        # output projection
        self.or_ = nn.Linear(embed_dim, embed_dim)
        self.oi_ = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        qr = self.qr(x)
        qi = self.qi(x)
        kr = self.kr(x)
        ki = self.ki(x)
        vr = self.vr(x)
        vi = self.vi(x)

        def split_heads(t):
            return t.reshape(B, N, H, D).permute(0, 2, 1, 3)

        qr, qi = split_heads(qr), split_heads(qi)
        kr, ki = split_heads(kr), split_heads(ki)
        vr, vi = split_heads(vr), split_heads(vi)

        # Transpose K => (B,H,D,N)
        kr = kr.transpose(-1, -2)
        ki = ki.transpose(-1, -2)

        def complex_matmul(a_r, a_i, b_r, b_i, conj_b=False):
            if conj_b:
                b_i = -b_i
            out_r = a_r @ b_r - a_i @ b_i
            out_i = a_r @ b_i + a_i @ b_r
            return out_r, out_i

        # Q x conj(K^T) => (B,H,N,N)
        scores_r, scores_i = complex_matmul(qr, qi, kr, ki, conj_b=True)
        scale = 1.0 / (D ** 0.5)
        scores_r, scores_i = scores_r * scale, scores_i * scale

        # magnitude => softmax => (B,H,N,N)
        attn_mag = torch.sqrt(scores_r**2 + scores_i**2)
        attn_weights = F.softmax(attn_mag, dim=-1)

        # Weighted sum => (B,H,N,D)
        def complex_matmul2(a_r, a_i, b_r, b_i):
            return a_r @ b_r - a_i @ b_i, a_r @ b_i + a_i @ b_r

        out_r, out_i = complex_matmul2(
            attn_weights, torch.zeros_like(attn_weights),
            vr, vi
        )

        def merge_heads(rr, ii):
            rr = rr.permute(0, 2, 1, 3).reshape(B, N, C)
            ii = ii.permute(0, 2, 1, 3).reshape(B, N, C)
            return rr, ii

        out_r, out_i = merge_heads(out_r, out_i)
        fr = self.or_(out_r) - self.oi_(out_i)
        fi = self.or_(out_i) + self.oi_(out_r)
        out = fr + 0.1 * fi
        return out

###############################################################################
# Minimal quantum backbone 
###############################################################################
class QuantumBackbone(nn.Module):
    """
    (B,P,3) => multiple MHA => single scalar or 9 scalar. 
    We'll do a simple in_dim=3 => embed => out_dim => e.g. 1 or 9
    """
    def __init__(self, in_dim=3, out_dim=1, embed_dim=16, num_heads=2, num_layers=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.input_linear = nn.Linear(in_dim, embed_dim)
        self.layers = nn.ModuleList([
            QuantumMultiHeadSelfAttention(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(embed_dim, out_dim)

    def forward(self, feats):
        # feats => (B,P,in_dim=3)
        B, P, C = feats.shape
        x = self.input_linear(feats)
        for layer in self.layers:
            x = layer(x)
        x = self.output_linear(x)
        return x

###############################################################################
# Our final "QuantumManiModel" => pos_net => 1, ori_net => 9
###############################################################################
class QuantumManiModel(nn.Module):
    def __init__(self, voxelize=True, voxel_size=0.01, radius_threshold=0.12, feature_point_radius=0.02):
        super().__init__()
        self.pos_net = QuantumBackbone(in_dim=3, out_dim=1, embed_dim=16, num_heads=2, num_layers=2)
        self.ori_net = QuantumBackbone(in_dim=3, out_dim=9, embed_dim=16, num_heads=2, num_layers=2)
        self.feature_point_radius = feature_point_radius

    def forward(self, inputs, 
                train_pos=False,
                reference_point=None,
                distance_threshold=0.3,
                random_drop=False,
                mask_part=False):
        """
        replicate your logic from 'Model_mani':
         1) seg_pointcloud if reference_point => new xyz, feats
         2) random_dropout, mask_part if needed
         3) pos_net => single scalar => weighted average => pos
         4) ori_net => 9 => local average => orientation
         5) avoid in-place modifications for orientation => we do a new tensor
        """
        xyz_in = inputs["xyz"]
        rgb_in = inputs["rgb"]
        B = xyz_in.shape[0]

        new_xyz = []
        new_feats = []
        for i in range(B):
            if reference_point is not None:
                data = seg_pointcloud(xyz_in[i], rgb_in[i], reference_point[i], distance=distance_threshold)
            else:
                data = {"xyz": xyz_in[i], "rgb": rgb_in[i]}
            if random_drop:
                data = random_dropout(data["xyz"], data["rgb"])
            if mask_part:
                data = mask_part_point_cloud(data["xyz"], data["rgb"])
            new_xyz.append(data["xyz"])
            new_feats.append(data["rgb"])

        new_xyz = torch.stack(new_xyz, dim=0)    # => (B,P',3)
        new_feats = torch.stack(new_feats, dim=0)# => (B,P',3)

        # pos_net => (B,P',1)
        heat = self.pos_net(new_feats)  
        heat = heat.squeeze(-1)          # (B,P')
        w = F.softmax(heat, dim=-1)      # => (B,P')
        pos_pred = torch.sum(new_xyz * w.unsqueeze(-1), dim=1)  # (B,3)

        # ori_net => (B,P',9)
        feats_ori = self.ori_net(new_feats)  # (B,P',9)
        B, Pp, O = feats_ori.shape

        # local average near pos_pred => replicate your logic, but no in-place
        out_ori = []
        for i in range(B):
            data2 = seg_pointcloud(new_xyz[i], new_xyz[i], pos_pred[i],
                                   distance=self.feature_point_radius,
                                   extra_data={"feature": feats_ori[i]})
            if data2["xyz"].shape[0] == 0:
                # fallback => average
                out_ori.append(feats_ori[i].mean(dim=0))
            else:
                out_ori.append(data2["feature"].mean(dim=0))
        out_ori = torch.stack(out_ori, dim=0)  # => (B,9)

        # reshape => (B,3,3), normalize each row in a new tensor
        out_ori_reshaped = out_ori.view(B,3,3)
        # DO NOT do in-place => create new rows
        row0 = out_ori_reshaped[:, 0, :]
        row1 = out_ori_reshaped[:, 1, :]
        row2 = out_ori_reshaped[:, 2, :]

        # normalize each row separately
        row0 = row0 / (row0.norm(dim=-1, keepdim=True) + 1e-8)
        row1 = row1 / (row1.norm(dim=-1, keepdim=True) + 1e-8)
        row2 = row2 / (row2.norm(dim=-1, keepdim=True) + 1e-8)

        # combine => (B,3,3)
        new_ori = torch.stack([row0, row1, row2], dim=1)
        new_ori = new_ori.reshape(B, 9)

        return pos_pred, new_ori

###############################################################################
# Now replicate the training logic with your SE3Demo etc.
###############################################################################
def main():
    # load config
    from omegaconf import OmegaConf
    all_cfg = OmegaConf.load("config/mug/pick/config.json")
    cfg = all_cfg.mani
    cfg_seg = all_cfg.seg

    wd = os.path.join("experiments", "mug", "pick")
    os.makedirs(wd, exist_ok=True)
    demo_path = os.path.join("data", "mug", "pick", "demos.npz")

    # 1) load dataset
    from utils.data_utils import SE3Demo
    demo = SE3Demo(demo_path, data_aug=cfg.data_aug, aug_methods=cfg.aug_methods, device="cpu")
    train_size = int(len(demo)*cfg.train_demo_ratio)
    test_size = len(demo)-train_size
    from torch.utils.data import random_split, DataLoader
    train_dataset, test_dataset = random_split(demo, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

    model = QuantumManiModel(
        voxelize=True, 
        voxel_size=0.01, 
        radius_threshold=0.12,
        feature_point_radius=0.02
    ).to("cpu")

    optm = torch.optim.Adam(model.parameters(), lr=0.001)
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optm, step_size=int(cfg.epoch/5), gamma=0.5)
    loss_fn = nn.MSELoss()
    best_test_loss = float('inf')

    from utils.loss_utils import double_geodesic_distance_between_poses

    for epoch in range(cfg.epoch):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epoch}")
        model.train()

        for i, data in enumerate(progress_bar):
            # data => { "xyz":(B,P,3), "rgb":(B,P,3), "seg_center":(B,3), "axes":(B,9), ...}
            xyz_b = data["xyz"]
            rgb_b = data["rgb"]
            seg_center_b = data["seg_center"]
            axes_b = data["axes"]  # shape => (B,9)

            optm.zero_grad()

            # replicate your "ref_point" logic
            with torch.no_grad():
                ref_pos, ref_dir = model(
                    {"xyz": xyz_b, "rgb": rgb_b},
                    random_drop=False,
                    mask_part=cfg.mask_part
                )
            if cfg.ref_point == "seg_net":
                training_ref_point = ref_pos
            elif cfg.ref_point == "gt":
                training_ref_point = seg_center_b
            else:
                training_ref_point = None

            out_pos, out_dir = model(
                {"xyz": xyz_b, "rgb": rgb_b},
                train_pos=True,
                reference_point=training_ref_point,
                distance_threshold=cfg.distance_threshold,
                random_drop=cfg.random_drop,
                mask_part=cfg.mask_part
            )

            pos_loss = loss_fn(out_pos, seg_center_b)
            ori_loss = loss_fn(out_dir, axes_b)
            if epoch < cfg.pos_warmup_epoch:
                total_loss = pos_loss
            else:
                total_loss = pos_loss + 0.1 * ori_loss

            total_loss.backward()
            optm.step()

            # geodesic check
            Bsz = axes_b.shape[0]
            T1 = torch.zeros([Bsz,4,4])
            T2 = torch.zeros_like(T1)
            T1[:, :3, :3] = axes_b.reshape(Bsz,3,3).transpose(1,2)
            T1[:, :3, 3] = seg_center_b
            T1[:, 3, 3] = 1.
            T2[:, :3, :3] = out_dir.reshape(Bsz,3,3).transpose(1,2)
            T2[:, :3, 3] = out_pos
            T2[:, 3, 3] = 1.
            t_loss, r_loss = double_geodesic_distance_between_poses(T1, T2, return_both=True)
            progress_bar.set_postfix(pos_loss=t_loss.item(), ori_loss=r_loss.item())

        # validation
        model.eval()
        test_pos_loss = 0.0
        test_ori_loss = 0.0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                xyz_b = data["xyz"]
                rgb_b = data["rgb"]
                seg_center_b = data["seg_center"]
                axes_b = data["axes"]

                out_pos, out_dir = model(
                    {"xyz": xyz_b, "rgb": rgb_b},
                    train_pos=False,
                    reference_point=seg_center_b,
                    distance_threshold=cfg.distance_threshold,
                    random_drop=False,
                    mask_part=cfg.mask_part
                )
                # compute MSE
                p_loss = loss_fn(out_pos, seg_center_b).item()
                o_loss = loss_fn(out_dir, axes_b).item()
                test_pos_loss += p_loss
                test_ori_loss += o_loss

            test_pos_loss /= len(test_loader)
            test_ori_loss /= len(test_loader)
            print(f"Epoch {epoch}: test pos loss={test_pos_loss}, test ori loss={test_ori_loss}")
            if test_pos_loss + test_ori_loss < best_test_loss:
                best_test_loss = test_pos_loss + test_ori_loss
                torch.save(model.state_dict(), os.path.join("experiments","mug","pick","quantum_mani_best.pth"))
                print("Model saved!")

        scheduler.step()

if __name__ == "__main__":
    main()
