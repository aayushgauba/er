import os
import sys
sys.path.append(".")
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.axangles import axangle2mat, mat2axangle
from copy import deepcopy

def calculate_norm_loss(output_directions):
    # calculate ||R^T*R - R_trace||_F
    R = output_directions.reshape(-1, 3, 3)
    R = torch.bmm(R.permute(0, 2, 1), R)    # column-wise
    norm_loss = (torch.bmm(R.permute(0, 2, 1), R) - torch.eye(3).repeat(R.shape[0], 1, 1).to(output_directions.device)).norm()

    return norm_loss

def R_to_phi(R):
    bs = R.shape[0]
    phi = torch.zeros(bs, 3).to(R.device)
    phi[:, 0] = R[:, 2, 1]
    phi[:, 1] = R[:, 0, 2]
    phi[:, 2] = R[:, 1, 0]

    return phi
    
def geodesic_distance_between_R(R1, R2):
    R1_T = R1.transpose(1, 2)
    R = torch.einsum("bmn,bnk->bmk", R1_T, R2)
    diagonals = torch.diagonal(R, dim1=1, dim2=2)
    traces = torch.sum(diagonals, dim=1).unsqueeze(1)   # [bs]
    dist = torch.abs(torch.arccos((traces - 1)/2))

    return dist

def double_geodesic_distance_between_poses(T1, T2, return_both=False):
    R_1, t_1 = T1[:, :3, :3], T1[:, :3, 3]
    R_2, t_2 = T2[:, :3, :3], T2[:, :3, 3]

    dist_R_square = geodesic_distance_between_R(R_1, R_2) ** 2
    dist_t_square = torch.sum((t_1-t_2) ** 2, dim=1)
    dist = torch.sqrt(dist_R_square.squeeze(-1) + dist_t_square)    # [bs]

    if return_both:
        return torch.sqrt(dist_t_square).mean(), torch.sqrt(dist_R_square).mean()
    else:
        return dist.mean()
    
class SE3Demo(Dataset):
    def __init__(self, demo_dir, data_aug=False, device="cpu", aug_methods=[]):
        super(SE3Demo, self).__init__()

        self.device = device
        demo = np.load(demo_dir, allow_pickle=True, mmap_mode='r') 
        if isinstance(demo, np.ndarray):
            demo = demo.item()
        traj_num, video_len, point_num, _ = demo["xyz"].shape

        self.xyz = torch.from_numpy(demo["xyz"][:, 0:20, ...]).float().to(self.device).reshape(-1, point_num, 3)
        self.rgb = torch.from_numpy(demo["rgb"][:, 0:20, ...]).float().to(self.device).reshape(-1, point_num, 3)
        self.seg_center = torch.from_numpy(demo["seg_center"][:, 0:20, ...]).float().to(self.device).reshape(-1, 3)
        self.axes = torch.from_numpy(demo["axes"][:, 0:20, ...]).float().to(self.device).reshape(-1, 9)

        self.data_aug = data_aug
        self.aug_methods = aug_methods

    def __len__(self):
        return self.xyz.shape[0]

    def __getitem__(self, index):
        data = {
            "xyz": self.xyz[index],
            "rgb": self.rgb[index],
            "seg_center": self.seg_center[index],
            "axes": self.axes[index],
        }

        with torch.no_grad():
            if self.data_aug:
                for method in self.aug_methods:
                    data = globals()[method](deepcopy(data))

        return data

def seg_pointcould_with_boundary(xyz, rgb, xyz_for_seg, x_min, x_max, y_min, y_max, z_min, z_max):
    rgb = rgb[
        (xyz_for_seg[:, 0] >= x_min) & (xyz_for_seg[:, 0] <= x_max) &
        (xyz_for_seg[:, 1] >= y_min) & (xyz_for_seg[:, 1] <= y_max) &
        (xyz_for_seg[:, 2] >= z_min) & (xyz_for_seg[:, 2] <= z_max)
    ]
    xyz = xyz[
        (xyz_for_seg[:, 0] >= x_min) & (xyz_for_seg[:, 0] <= x_max) &
        (xyz_for_seg[:, 1] >= y_min) & (xyz_for_seg[:, 1] <= y_max) &
        (xyz_for_seg[:, 2] >= z_min) & (xyz_for_seg[:, 2] <= z_max)
    ]
    return {
        "xyz": xyz,
        "rgb": rgb
    }

def seg_pointcloud(xyz, rgb, reference_point, distance=0.3, extra_data=None):
    distances = torch.norm(xyz - reference_point, dim=1)
    xyz = xyz[distances < distance]
    rgb = rgb[distances < distance]

    data = {
        "xyz": xyz,
        "rgb": rgb,
    }

    if extra_data != None:
        for k in list(extra_data.keys()):
            if k == "xyz" or k == "rgb":
                continue
            data[k] = extra_data[k][distances < distance]
            if torch.isnan(data[k]).any():
                print("Nan detected!")
    return data


def dropout_to_certain_number(xyz, rgb, target_num):
    current_num = xyz.shape[0]
    assert target_num < current_num
    random_indices = torch.randperm(xyz.shape[0])[:target_num]

    xyz = xyz[random_indices]
    rgb = rgb[random_indices]

    return {
        "xyz": xyz,
        "rgb": rgb,
    }


def random_dropout(xyz, rgb, remain_point_ratio=[0.5, 0.9]):
    desired_points = int(np.random.uniform(remain_point_ratio[0], remain_point_ratio[1]) * xyz.shape[0])
    random_indices = torch.randperm(xyz.shape[0])[:desired_points]

    xyz = xyz[random_indices]
    rgb = rgb[random_indices]

    return {
        "xyz": xyz,
        "rgb": rgb,
    }

def downsample_table(data, reference_rgb=[0.1176, 0.4392, 0.4078], total_points=2500):
    xyz = data["xyz"]
    rgb = data["rgb"]
    table_mask = (rgb - torch.tensor(reference_rgb).to(xyz.device)).norm(dim=1) < 0.1
    table_xyz = xyz[table_mask]
    table_rgb = rgb[table_mask]

    non_table_xyz = xyz[~table_mask]
    non_table_rgb = rgb[~table_mask]

    if non_table_xyz.shape[0] <= total_points:
        desired_points = total_points - non_table_xyz.shape[0]
        random_indices = torch.randperm(table_xyz.shape[0])[:desired_points]

        xyz = torch.cat([table_xyz[random_indices], non_table_xyz], dim=0)
        rgb = torch.cat([table_rgb[random_indices], non_table_rgb], dim=0)
    else:   # all from objects
        random_indices = torch.randperm(non_table_xyz.shape[0])[:total_points]

        xyz = non_table_xyz[random_indices]
        rgb = non_table_rgb[random_indices]

    data["xyz"] = xyz
    data["rgb"] = rgb
    return data

def jitter(data, std=0.03):
    data["xyz"] = data["xyz"] + std * torch.randn(data["xyz"].shape).to(data["xyz"].device)
    return data

def random_dropping_color(data, drop_ratio=0.3):
    # randomly remove some points' RGB to [0,0,0]
    N = data["xyz"].shape[0]
    mask = np.random.choice([0, 1], size=N, replace=True, p=[1-drop_ratio, drop_ratio])
    data["rgb"][np.where(mask)] = torch.tensor([0., 0., 0.]).to(data["rgb"].device)
    
    return data

def color_jitter(data, std=0.005):
    data["rgb"] = torch.clamp(data["rgb"] + (torch.rand(data["rgb"].shape).to(data["xyz"].device) - 0.5) * 2 * std, 0, 1)
    return data

def zero_color(data):
    # remove all color
    data["rgb"] = torch.zeros_like(data["rgb"])
    return data

from torchvision.transforms import functional as TF
def hsv_transform(data, hue_shift_range=[-0.4, 0.4], sat_shift_range=[0.5, 1.5], val_shift_range=[0.5, 2]):
    img_rgb = data["rgb"].T.unsqueeze(-1) # [N, 3] -> [3, N] -> [3, N, 1], and the adjust functions requires [3, H, W]

    hue_shift = np.random.random_sample() * (hue_shift_range[1] - hue_shift_range[0]) + hue_shift_range[0]
    sat_shift = np.random.random_sample() * (sat_shift_range[1] - sat_shift_range[0]) + sat_shift_range[0]
    val_shift = np.random.random_sample() * (val_shift_range[1] - val_shift_range[0]) + val_shift_range[0]

    img_rgb = TF.adjust_hue(img_rgb, hue_factor=hue_shift)
    img_rgb = TF.adjust_saturation(img_rgb, saturation_factor=sat_shift)
    img_rgb = TF.adjust_brightness(img_rgb, brightness_factor=val_shift)

    data["rgb"] = img_rgb.squeeze(-1).T

    return data

import potpourri3d as pp3d
def geodesic_distance_from_pcd(point_cloud, keypoint_index):
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()
    solver = pp3d.PointCloudHeatSolver(point_cloud)

    # Compute the geodesic distance to point 4
    dists = solver.compute_distance(keypoint_index)

    return torch.from_numpy(dists).float()

def get_heatmap(point_cloud, keypoint_index, distance="geodesic", max_value = 10.0, std_dev=0.005):
    # distance: "l2" or "geodesic"

    # Extract keypoint coordinates
    keypoint = point_cloud[keypoint_index]
    if distance == "l2":
        # Compute the L2 distance from the keypoint to all other points
        distances = torch.norm(point_cloud - keypoint, dim=1)
    elif distance == "geodesic":
        # Compute the geodesic distance from the keypoint to all other points
        distances = geodesic_distance_from_pcd(point_cloud, keypoint_index)
    heatmap_values = torch.exp(-0.5 * (distances / std_dev) ** 2)
    heatmap_values /= torch.max(heatmap_values)

    heatmap_values *= max_value

    return heatmap_values

import random

def mask_part_point_cloud(xyz, rgb, mask_radius=0.015):
    N, _ = xyz.shape

    center_idx = random.randint(0, N-1)
    center_point = xyz[center_idx]

    distances = torch.sqrt(torch.sum((xyz - center_point) ** 2, dim=1))
    mask = distances < mask_radius

    masked_point_cloud = xyz[~mask]
    masked_rgb = rgb[~mask]

    return {
        "xyz": masked_point_cloud,
        "rgb": masked_rgb,
    }

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
        attn_weights = F.softmax(scores_mag, dim=-1)
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
