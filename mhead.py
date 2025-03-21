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
from utils.data_utils import SE3Demo, seg_pointcloud, random_dropout, mask_part_point_cloud
from utils.loss_utils import double_geodesic_distance_between_poses


class UnitaryComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        assert in_features == out_features, "in_features must equal out_features for unitarity."
        self.in_features = in_features
        self.out_features = out_features
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_complex = self.weight_real + 1j * self.weight_imag  # (out_features, in_features)
        Q, _ = torch.linalg.qr(weight_complex)
        out = x @ Q.conj().T  # x shape (..., in_features)
        return out


class UnitaryComplexLinearWrapper(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.unitary = UnitaryComplexLinear(in_features, out_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_complex = x.to(torch.complex64)
        out_complex = self.unitary(x_complex)
        return out_complex


class QuantumPhaseGate(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase_shift = torch.exp(1j * self.theta)  
        return x * phase_shift


class QuantumMultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim: int = 16, num_heads: int = 2):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.linear_q = UnitaryComplexLinearWrapper(embed_dim, embed_dim)
        self.linear_k = UnitaryComplexLinearWrapper(embed_dim, embed_dim)
        self.linear_v = UnitaryComplexLinearWrapper(embed_dim, embed_dim)
        self.linear_out = UnitaryComplexLinearWrapper(embed_dim, embed_dim)
        self.amplification = nn.Parameter(torch.tensor(1.0))
        self.phase_gate = QuantumPhaseGate(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        q = q.view(B, N, H, D).permute(0, 2, 1, 3)
        k = k.view(B, N, H, D).permute(0, 2, 1, 3)
        v = v.view(B, N, H, D).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.conj().transpose(-1, -2)) / (D ** 0.5)
        scores_mag = torch.abs(scores) * self.amplification
        attn_weights = F.softmax(scores_mag, dim=-1)  # real tensor (B, H, N, N)
        attn_weights = attn_weights.to(dtype=v.dtype)  # cast to complex type
        out = torch.matmul(attn_weights, v)  # (B, H, N, D) complex
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.embed_dim)
        out = self.phase_gate(out)
        out_complex = self.linear_out(out)  # complex tensor
        out_real = out_complex.real + 0.1 * out_complex.imag
        return out_real


class QuantumBackbone(nn.Module):

    def __init__(self, in_dim: int = 3, out_dim: int = 1, embed_dim: int = 16,
                 num_heads: int = 2, num_layers: int = 2):
        super().__init__()
        self.input_linear = nn.Linear(in_dim, embed_dim)
        self.layers = nn.ModuleList([
            QuantumMultiHeadSelfAttention(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(embed_dim, out_dim)
        
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.input_linear(feats)
        for layer in self.layers:
            x = layer(x)
        x = self.output_linear(x)
        return x


class QuantumManiModel(nn.Module):

    def __init__(self, voxelize: bool = True, voxel_size: float = 0.01,
                 radius_threshold: float = 0.12, feature_point_radius: float = 0.02):
        super().__init__()
        self.pos_net = QuantumBackbone(in_dim=3, out_dim=1, embed_dim=16, num_heads=2, num_layers=2)
        self.ori_net = QuantumBackbone(in_dim=3, out_dim=9, embed_dim=16, num_heads=2, num_layers=2)
        self.feature_point_radius = feature_point_radius

    def forward(self, inputs: dict, train_pos: bool = False,
                reference_point: torch.Tensor = None, distance_threshold: float = 0.3,
                random_drop: bool = False, mask_part: bool = False):
        xyz_in = inputs["xyz"]
        rgb_in = inputs["rgb"]
        B = xyz_in.shape[0]

        new_xyz, new_feats = [], []
        for i in range(B):
            if reference_point is not None:
                data = seg_pointcloud(xyz_in[i], rgb_in[i], reference_point[i],
                                      distance=distance_threshold)
            else:
                data = {"xyz": xyz_in[i], "rgb": rgb_in[i]}
            if random_drop:
                data = random_dropout(data["xyz"], data["rgb"])
            if mask_part:
                data = mask_part_point_cloud(data["xyz"], data["rgb"])
            new_xyz.append(data["xyz"])
            new_feats.append(data["rgb"])

        new_xyz = torch.stack(new_xyz, dim=0)
        new_feats = torch.stack(new_feats, dim=0)

        heat = self.pos_net(new_feats).squeeze(-1)
        w = F.softmax(heat, dim=-1)
        pos_pred = torch.sum(new_xyz * w.unsqueeze(-1), dim=1)
        feats_ori = self.ori_net(new_feats)
        out_ori = []
        for i in range(B):
            data2 = seg_pointcloud(new_xyz[i], new_xyz[i], pos_pred[i],
                                   distance=self.feature_point_radius,
                                   extra_data={"feature": feats_ori[i]})
            out_ori.append(feats_ori[i].mean(dim=0) if data2["xyz"].shape[0] == 0 
                           else data2["feature"].mean(dim=0))
        out_ori = torch.stack(out_ori, dim=0)
        out_ori_reshaped = out_ori.view(B, 3, 3)
        row0 = out_ori_reshaped[:, 0, :] / (out_ori_reshaped[:, 0, :].norm(dim=-1, keepdim=True) + 1e-8)
        row1 = out_ori_reshaped[:, 1, :] / (out_ori_reshaped[:, 1, :].norm(dim=-1, keepdim=True) + 1e-8)
        row2 = out_ori_reshaped[:, 2, :] / (out_ori_reshaped[:, 2, :].norm(dim=-1, keepdim=True) + 1e-8)
        new_ori = torch.stack([row0, row1, row2], dim=1).reshape(B, 9)
        return pos_pred, new_ori


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_cfg = OmegaConf.load("config/mug/pick/config.json")
    cfg = all_cfg.mani
    cfg_seg = all_cfg.seg

    wd = os.path.join("experiments", "mug", "pick")
    os.makedirs(wd, exist_ok=True)
    demo_path = os.path.join("data", "mug", "pick", "demo.npz")

    demo = SE3Demo(demo_path, data_aug=cfg.data_aug, aug_methods=cfg.aug_methods, device=str(device))
    train_size = int(len(demo) * cfg.train_demo_ratio)
    test_size = len(demo) - train_size
    train_dataset, test_dataset = random_split(demo, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

    model = QuantumManiModel(
        voxelize=True,
        voxel_size=0.01,
        radius_threshold=0.12,
        feature_point_radius=0.02
    ).to(device)

    optm = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optm, step_size=int(cfg.epoch / 5), gamma=0.5)
    loss_fn = nn.MSELoss()
    best_test_loss = float("inf")

    for epoch in range(cfg.epoch):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epoch}")
        model.train()
        for data in progress_bar:
            xyz_b = data["xyz"].to(device)
            rgb_b = data["rgb"].to(device)
            seg_center_b = data["seg_center"].to(device)
            axes_b = data["axes"].to(device)

            optm.zero_grad()
            with torch.no_grad():
                ref_pos, _ = model({"xyz": xyz_b, "rgb": rgb_b},
                                   random_drop=False,
                                   mask_part=cfg.mask_part)
            training_ref_point = (ref_pos if cfg.ref_point == "seg_net"
                                  else seg_center_b if cfg.ref_point == "gt"
                                  else None)

            out_pos, out_dir = model({"xyz": xyz_b, "rgb": rgb_b},
                                     train_pos=True,
                                     reference_point=training_ref_point,
                                     distance_threshold=cfg.distance_threshold,
                                     random_drop=cfg.random_drop,
                                     mask_part=cfg.mask_part)

            pos_loss = loss_fn(out_pos, seg_center_b)
            ori_loss = loss_fn(out_dir, axes_b)
            total_loss = pos_loss if epoch < cfg.pos_warmup_epoch else pos_loss + 0.1 * ori_loss

            total_loss.backward()
            optm.step()
            Bsz = axes_b.shape[0]
            T1 = torch.zeros([Bsz, 4, 4], device=device)
            T2 = torch.zeros_like(T1)
            T1[:, :3, :3] = axes_b.reshape(Bsz, 3, 3).transpose(1, 2)
            T1[:, :3, 3] = seg_center_b
            T1[:, 3, 3] = 1.0
            T2[:, :3, :3] = out_dir.reshape(Bsz, 3, 3).transpose(1, 2)
            T2[:, :3, 3] = out_pos
            T2[:, 3, 3] = 1.0
            t_loss, r_loss = double_geodesic_distance_between_poses(T1, T2, return_both=True)
            progress_bar.set_postfix(pos_loss=t_loss.item(), ori_loss=r_loss.item())

        model.eval()
        test_pos_loss, test_ori_loss = 0.0, 0.0
        with torch.no_grad():
            for data in test_loader:
                xyz_b = data["xyz"].to(device)
                rgb_b = data["rgb"].to(device)
                seg_center_b = data["seg_center"].to(device)
                axes_b = data["axes"].to(device)

                out_pos, out_dir = model({"xyz": xyz_b, "rgb": rgb_b},
                                         train_pos=False,
                                         reference_point=seg_center_b,
                                         distance_threshold=cfg.distance_threshold,
                                         random_drop=False,
                                         mask_part=cfg.mask_part)
                test_pos_loss += loss_fn(out_pos, seg_center_b).item()
                test_ori_loss += loss_fn(out_dir, axes_b).item()

            test_pos_loss /= len(test_loader)
            test_ori_loss /= len(test_loader)
            print(f"Epoch {epoch}: test pos loss={test_pos_loss}, test ori loss={test_ori_loss}")
            if test_pos_loss + test_ori_loss < best_test_loss:
                best_test_loss = test_pos_loss + test_ori_loss
                torch.save(model.state_dict(), os.path.join("experiments", "mug", "pick", "quantum_mani_best.pth"))
                print("Model saved!")
        scheduler.step()


if __name__ == "__main__":
    main()
