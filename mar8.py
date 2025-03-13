import os
import sys
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from networks import *
from networks.se3_backbone import SE3Backbone
from networks.se3_transformer.model.fiber import Fiber
from utils.data_utils import SE3Demo, seg_pointcloud, random_dropout, mask_part_point_cloud
from utils.vis import save_pcd_as_pcd
from utils.loss_utils import double_geodesic_distance_between_poses
from omegaconf import OmegaConf

# Load configuration
all_cfg = OmegaConf.load("config/mug/pick/config.json")
cfg = all_cfg.mani
cfg_seg = all_cfg.seg

# Set working directories
wd = os.path.join("experiments", "mug", "pick")
os.makedirs(wd, exist_ok=True)
demo_path = os.path.join("data", "mug", "pick", "demos.npz")

demo = SE3Demo(demo_path, data_aug=cfg.data_aug, aug_methods=cfg.aug_methods, device="cpu")
train_size = int(len(demo) * cfg.train_demo_ratio)
test_size = len(demo) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(demo, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=True)

class Model(nn.Module):
    def __init__(self, voxelize=True, voxel_size=0.01, radius_threshold=0.12, feature_point_radius=0.02):
        super().__init__()
        self.pos_net = SE3Backbone(
            fiber_out=Fiber({"0": 1}),
            num_layers=4,
            num_degrees=3,
            num_channels=8,
            num_heads=1,
            channels_div=2,
            voxelize=voxelize,
            voxel_size=voxel_size,
            radius_threshold=radius_threshold,
        )
        self.ori_net = SE3Backbone(
            fiber_out=Fiber({"1": 3}),
            num_layers=4,
            num_degrees=4,
            num_channels=8,
            num_heads=1,
            channels_div=2,
            voxelize=voxelize,
            voxel_size=voxel_size,
            radius_threshold=radius_threshold,
        )
        self.feature_point_radius = feature_point_radius

    def forward(self, inputs, train_pos=False, reference_point=None, distance_threshold=0.3, random_drop=False, mask_part=False):
        bs = inputs["xyz"].shape[0]
        new_inputs = {"xyz": [], "rgb": [], "feature": []}
        for i in range(bs):
            data = seg_pointcloud(inputs["xyz"][i], inputs["rgb"][i], reference_point[i], distance=distance_threshold) if reference_point is not None else {"xyz": inputs["xyz"][i], "rgb": inputs["rgb"][i]}
            if random_drop:
                data = random_dropout(data["xyz"], data["rgb"])
            if mask_part:
                data = mask_part_point_cloud(data["xyz"], data["rgb"])
            new_inputs["xyz"].append(data["xyz"])
            new_inputs["rgb"].append(data["rgb"])
            new_inputs["feature"].append(new_inputs["rgb"][i])
        inputs = new_inputs

        seg_output = self.pos_net(inputs)
        xyz = seg_output["xyz"]
        feature = seg_output["feature"]
        pos_weights = []
        output_pos = torch.zeros([len(xyz), 3]).to("cpu")
        for i in range(len(xyz)):
            pos_weight = torch.nn.functional.softmax(feature[i].reshape(-1, 1), dim=0).squeeze()
            output_pos[i] = (xyz[i].T * pos_weight).T.sum(dim=0)
            pos_weights.append(pos_weight)
        return output_pos

model = Model().to("cpu")
optm = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optm, step_size=int(cfg.epoch/5), gamma=0.5)
loss_fn = torch.nn.MSELoss()
best_test_loss = float("inf")

# Training loop
for epoch in range(cfg.epoch):
    model.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epoch}")
    for i, data in enumerate(progress_bar):
        optm.zero_grad()
        output_pos = model({"xyz": data["xyz"], "rgb": data["rgb"]}, random_drop=cfg.random_drop, mask_part=cfg.mask_part)
        loss = loss_fn(output_pos, data["seg_center"])
        loss.backward()
        optm.step()
        progress_bar.set_postfix(loss=loss.item())

    model.eval()
    with torch.no_grad():
        test_loss = sum(loss_fn(model({"xyz": data["xyz"], "rgb": data["rgb"]}), data["seg_center"]).item() for data in test_loader) / len(test_loader)
        print(f"Epoch: {epoch}, Test Loss: {test_loss}")
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(wd, "model.pth"))
            print("Model saved!")
    scheduler.step()
