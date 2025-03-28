# %% [markdown]
# ## Start here ##

# %%
import os
import sys
sys.path.append(".")
import torch
from networks import *
from omegaconf import OmegaConf
import os
from torch.utils.data import DataLoader
from utils.data_utils import SE3Demo
from utils.loss_utils import double_geodesic_distance_between_poses
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import StepLR

# %%
import logging
import sys

# Configure logging: This will log both to the console and to a file.
logging.basicConfig(
    level=logging.INFO,  # Set this to DEBUG for more detailed output
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Output to console
        logging.FileHandler("training.log", mode="w")  # Write to training.log
    ]
)
all_cfg = OmegaConf.load(f"config/mug/pick/config.json")
cfg = all_cfg.mani
cfg_seg = all_cfg.seg

# %%
wd = os.path.join("experiments", "mug", "pick")
os.makedirs(wd, exist_ok=True)
# demo_path = os.path.join("data", "mug", "pick", "demos.npz")
demo_path = os.path.join("data", "mug", "pick", "demo.npz")

# %%
demo = SE3Demo(demo_path, data_aug=cfg.data_aug, aug_methods=cfg.aug_methods, device="cpu") 
#demo = SE3Demo(demo_path, data_aug=True, aug_methods=0, device='cuda')  # maybe change the config file sometime

# %%
cfg.data_aug, cfg.aug_methods

# %%
demo = SE3Demo(demo_path, data_aug=cfg.data_aug, aug_methods=cfg.aug_methods, device="cpu") 
#demo = SE3Demo(demo_path, data_aug=True, aug_methods=0, device='cuda')  # maybe change the config file sometime

# %%
train_size = int(len(demo) * cfg.train_demo_ratio)
test_size = len(demo) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(demo, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=True)

# %%


# %%
import torch.nn as nn
import torch.nn.functional as F

# %%
from networks.se3_backbone import SE3Backbone, ExtendedModule
from networks.se3_transformer.model.fiber import Fiber
from utils.data_utils import seg_pointcloud, random_dropout, mask_part_point_cloud

from utils.vis import save_pcd_as_pcd
from utils.data_utils import get_heatmap

# %%
class Model(nn.Module):
    def __init__(self, voxelize=True, voxel_size=0.01, radius_threshold=0.12, feature_point_radius=0.02):
        super().__init__()
        self.pos_net = SE3Backbone(
            fiber_out=Fiber({
                "0": 1, # one heatmap
            }),
            num_layers= 4,
            num_degrees= 3,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize,
            voxel_size= voxel_size,
            radius_threshold=radius_threshold,
        )

        self.ori_net = SE3Backbone(
            fiber_out=Fiber({
                "1": 3,
            }),
            num_layers= 4,
            num_degrees= 4,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize,
            voxel_size= voxel_size,
            radius_threshold=radius_threshold,
        )
        self.feature_point_radius = feature_point_radius

    def forward(self, inputs, train_pos=False, reference_point=None, distance_threshold=0.3, random_drop=False, draw_pcd=False, pcd_name=None, mask_part=False, save_ori_feature=False):
        bs = inputs["xyz"].shape[0]
        new_inputs = {
            "xyz": [],
            "rgb": [],
            "feature": []
        }
        gt_heatmaps = []
        for i in range(bs):
            if draw_pcd:
                os.makedirs("pcd/mani", exist_ok=True)
                distances = torch.norm(inputs["xyz"][i] - reference_point[i], dim=1)
                closest_point_idx = torch.argmin(distances)
                save_pcd_as_pcd(inputs["xyz"][i], inputs["rgb"][i], save_file=f"pcd/mani/original_{pcd_name}_{i}.pcd")

                gt_heatmaps.append(get_heatmap(inputs["xyz"][i], closest_point_idx, std_dev=0.015, max_value=1).to(self.pos_net.device))
                save_pcd_as_pcd(inputs["xyz"][i], gt_heatmaps[-1].unsqueeze(-1).repeat(1, 3)/torch.max(gt_heatmaps[-1]), save_file=f"pcd/mani/gt_heatmap_{pcd_name}_{i}.pcd", draw_heatmap=True)

            if reference_point != None:
                data = seg_pointcloud(inputs["xyz"][i], inputs["rgb"][i], reference_point[i], distance=distance_threshold)
            else:
                data = {
                    "xyz": inputs["xyz"][i],
                    "rgb": inputs["rgb"][i],
                }
            if random_drop:
                data = random_dropout(data["xyz"], data["rgb"])
            if mask_part:
                data = mask_part_point_cloud(data["xyz"], data["rgb"])
            new_inputs["xyz"].append(data["xyz"])
            new_inputs["rgb"].append(data["rgb"])
            new_inputs["feature"].append(new_inputs["rgb"][i])
        inputs = new_inputs

        # pos
        if train_pos:
            seg_output = self.pos_net(inputs)
            xyz = seg_output["xyz"]
            feature = seg_output["feature"]
            pos_weights = []

            output_pos = torch.zeros([len(xyz), 3]).to(self.device)
            for i in range(len(xyz)):
                if draw_pcd:
                    save_pcd_as_pcd(xyz[i], feature[i][:, 0].clone().unsqueeze(-1).repeat(1, 3)/torch.max(feature[i][:, 0].clone()), save_file=f"pcd/mani/pos_heatmap_{pcd_name}_{i}.pcd", draw_heatmap=True)

                pos_weight = torch.nn.functional.softmax(feature[i].reshape(-1, 1), dim=0).squeeze()
                output_pos[i] = (xyz[i].T * pos_weight).T.sum(dim=0)
                pos_weights.append(pos_weight)
        else:
            with torch.no_grad():
                seg_output = self.pos_net(inputs)
                xyz = seg_output["xyz"]
                feature = seg_output["feature"]
                pos_weights = []

#                output_pos = torch.zeros([len(xyz), 3]).to(self.device)
                output_pos = torch.zeros([len(xyz), 3]).to("cpu")
                for i in range(len(xyz)):
                    if draw_pcd:
                        save_pcd_as_pcd(xyz[i], feature[i][:, 0].clone().unsqueeze(-1).repeat(1, 3)/torch.max(feature[i][:, 0].clone()), save_file=f"pcd/mani/pos_heatmap_{pcd_name}_{i}.pcd", draw_heatmap=True)

                    pos_weight = torch.nn.functional.softmax(feature[i].reshape(-1, 1), dim=0).squeeze()
                    output_pos[i] = (xyz[i].T * pos_weight).T.sum(dim=0)
                    pos_weights.append(pos_weight)

        if draw_pcd:
            for i in range(len(xyz)):
                distances = torch.norm(xyz[i] - reference_point[i], dim=1)
                closest_point_idx = torch.argmin(distances)
                save_pcd_as_pcd(xyz[i], seg_output["given_graph"]["raw_node_feats"][i][:, :3], save_file=f"pcd/mani/ball_{pcd_name}_{i}.pcd")

        ori_output = self.ori_net(inputs)
        xyz = ori_output["xyz"]
        feature = ori_output["feature"]    # 3*3 = 9
#        output_ori = torch.zeros([len(xyz), 9]).to(self.device)
        output_ori = torch.zeros([len(xyz), 9]).to("cpu")

        if save_ori_feature:
            for i in range(len(xyz)):
                torch.save(feature[i].cpu(), f"pcd/mani/ori_feature_{pcd_name}_{i}.pt")

        for i in range(bs):
            newdata = seg_pointcloud(xyz[i], xyz[i], reference_point=output_pos[i], distance=self.feature_point_radius, extra_data={"feature": feature[i]})
            if newdata["xyz"].shape[0] == 0:
                # use the pos point
                output_ori[i] = (feature[i].T * pos_weights[i].detach()).T.sum(dim=0)
            else:
                output_ori[i] = newdata["feature"].mean(dim=0)

        for i in range(3):
            output_ori[:, 3*i:3*(i+1)] /= (torch.norm(output_ori[:, 3*i:3*(i+1)].clone(), dim=1).unsqueeze(1) + 1e-8)
        return output_pos, output_ori

# %%
model = Model()
model.to("cpu")

# %%
optm = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# %%
#policy = globals()[cfg.model](voxel_size=cfg.voxel_size, radius_threshold=cfg.radius_threshold).float().to(cfg.device)
#optm = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
scheduler = StepLR(optm, step_size=int(cfg.epoch/5), gamma=0.5)
loss_fn = torch.nn.MSELoss()


# %%
best_test_loss = 1e5

# %%
#torch.set_grad_enabled(True)

# %%
for epoch in range(cfg.epoch):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epoch}")
    model.train()

    logging.info("test")

    for i, data in enumerate(progress_bar):
        optm.zero_grad()

        output_pos = model(
            {"xyz": data["xyz"], "rgb": data["rgb"]}, 
            random_drop=cfg.random_drop,
            draw_pcd=cfg.draw_pcd,
            pcd_name=f"{i}",
            mask_part=cfg.mask_part,
        )

#        logging.info((output_pos[0][0]))
#        logging.info(data["seg_center"])
    
        loss = loss_fn(output_pos[0], data["seg_center"])
        logging.info(loss)
        logging.info(type(loss))
        loss.requires_grad = True
        loss.backward()
        optm.step()

        t_loss = torch.sqrt(torch.sum(torch.sqrt((output_pos[0]-data["seg_center"]) ** 2), dim=1)).mean()
        progress_bar.set_postfix(loss=t_loss.item())

    model.eval()

    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch_idx, data in enumerate(test_loader):
            output_pos = model(
                {"xyz": data["xyz"], "rgb": data["rgb"]}, 
                random_drop=False,
                draw_pcd=cfg.draw_pcd,
                pcd_name=f"test_{batch_idx}",
            )
            t_loss = torch.sqrt(torch.sum(torch.sqrt((output_pos[0]-data["seg_center"]) ** 2), dim=1)).mean()

            test_loss += t_loss.item()
        test_loss /= len(test_loader)
        logging.info(f"Epoch: , {epoch},  seg test loss: , {test_loss}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(wd, f"segnet.pth"))
            logging.info("Model saved!")

    scheduler.step()


# %% [markdown]
# ## train_mani ##

# %%
wd

# %%


# %%
all_cfg = OmegaConf.load(f"config/mug/pick/config.json")
cfg = all_cfg.mani
cfg_seg = all_cfg.seg

# %%
wd = os.path.join("experiments", "mug", "pick")
os.makedirs(wd, exist_ok=True)
demo_path = os.path.join("data", "mug", "pick", "demo.npz")

# %%


# %%
demo = SE3Demo(demo_path, data_aug=cfg.data_aug, aug_methods=cfg.aug_methods, device=cfg.device)
train_size = int(len(demo) * cfg.train_demo_ratio)
test_size = len(demo) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(demo, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=True)

# %%
class Model_mani(nn.Module):
    def __init__(self, voxelize=True, voxel_size=0.01, radius_threshold=0.12, feature_point_radius=0.02):
        super().__init__()
        self.pos_net = SE3Backbone(
            fiber_out=Fiber({
                "0": 1, # one heatmap
            }),
            num_layers= 4,
            num_degrees= 3,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize,
            voxel_size= voxel_size,
            radius_threshold=radius_threshold,
        )

        self.ori_net = SE3Backbone(
            fiber_out=Fiber({
                "1": 3,
            }),
            num_layers= 4,
            num_degrees= 4,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize,
            voxel_size= voxel_size,
            radius_threshold=radius_threshold,
        )
        self.feature_point_radius = feature_point_radius

    def forward(self, inputs, train_pos=False, reference_point=None, distance_threshold=0.3, random_drop=False, draw_pcd=False, pcd_name=None, mask_part=False, save_ori_feature=False):
        bs = inputs["xyz"].shape[0]
        new_inputs = {
            "xyz": [],
            "rgb": [],
            "feature": []
        }
        gt_heatmaps = []
        for i in range(bs):
            if draw_pcd:
                os.makedirs("pcd/mani", exist_ok=True)
                distances = torch.norm(inputs["xyz"][i] - reference_point[i], dim=1)
                closest_point_idx = torch.argmin(distances)
                save_pcd_as_pcd(inputs["xyz"][i], inputs["rgb"][i], save_file=f"pcd/mani/original_{pcd_name}_{i}.pcd")

#                gt_heatmaps.append(get_heatmap(inputs["xyz"][i], closest_point_idx, std_dev=0.015, max_value=1).to(self.pos_net.device))
                gt_heatmaps.append(get_heatmap(inputs["xyz"][i], closest_point_idx, std_dev=0.015, max_value=1).to("cuda"))
                save_pcd_as_pcd(inputs["xyz"][i], gt_heatmaps[-1].unsqueeze(-1).repeat(1, 3)/torch.max(gt_heatmaps[-1]), save_file=f"pcd/mani/gt_heatmap_{pcd_name}_{i}.pcd", draw_heatmap=True)

            if reference_point != None:
                data = seg_pointcloud(inputs["xyz"][i], inputs["rgb"][i], reference_point[i], distance=distance_threshold)
            else:
                data = {
                    "xyz": inputs["xyz"][i],
                    "rgb": inputs["rgb"][i],
                }
            if random_drop:
                data = random_dropout(data["xyz"], data["rgb"])
            if mask_part:
                data = mask_part_point_cloud(data["xyz"], data["rgb"])
            new_inputs["xyz"].append(data["xyz"])
            new_inputs["rgb"].append(data["rgb"])
            new_inputs["feature"].append(new_inputs["rgb"][i])
        inputs = new_inputs

        # pos
        if train_pos:
            seg_output = self.pos_net(inputs)
            xyz = seg_output["xyz"]
            feature = seg_output["feature"]
            pos_weights = []

            output_pos = torch.zeros([len(xyz), 3]).to("cuda")
            for i in range(len(xyz)):
                if draw_pcd:
                    save_pcd_as_pcd(xyz[i], feature[i][:, 0].clone().unsqueeze(-1).repeat(1, 3)/torch.max(feature[i][:, 0].clone()), save_file=f"pcd/mani/pos_heatmap_{pcd_name}_{i}.pcd", draw_heatmap=True)

                pos_weight = torch.nn.functional.softmax(feature[i].reshape(-1, 1), dim=0).squeeze()
                output_pos[i] = (xyz[i].T * pos_weight).T.sum(dim=0)
                pos_weights.append(pos_weight)
        else:
            with torch.no_grad():
                seg_output = self.pos_net(inputs)
                xyz = seg_output["xyz"]
                feature = seg_output["feature"]
                pos_weights = []

#                output_pos = torch.zeros([len(xyz), 3]).to(self.device)
                output_pos = torch.zeros([len(xyz), 3]).to("cuda")
                for i in range(len(xyz)):
                    if draw_pcd:
                        save_pcd_as_pcd(xyz[i], feature[i][:, 0].clone().unsqueeze(-1).repeat(1, 3)/torch.max(feature[i][:, 0].clone()), save_file=f"pcd/mani/pos_heatmap_{pcd_name}_{i}.pcd", draw_heatmap=True)

                    pos_weight = torch.nn.functional.softmax(feature[i].reshape(-1, 1), dim=0).squeeze()
                    output_pos[i] = (xyz[i].T * pos_weight).T.sum(dim=0)
                    pos_weights.append(pos_weight)

        if draw_pcd:
            for i in range(len(xyz)):
                distances = torch.norm(xyz[i] - reference_point[i], dim=1)
                closest_point_idx = torch.argmin(distances)
                save_pcd_as_pcd(xyz[i], seg_output["given_graph"]["raw_node_feats"][i][:, :3], save_file=f"pcd/mani/ball_{pcd_name}_{i}.pcd")

        ori_output = self.ori_net(inputs)
        xyz = ori_output["xyz"]
        feature = ori_output["feature"]    # 3*3 = 9
#        output_ori = torch.zeros([len(xyz), 9]).to(self.device)
        output_ori = torch.zeros([len(xyz), 9]).to("cuda")

        if save_ori_feature:
            for i in range(len(xyz)):
                torch.save(feature[i].cpu(), f"pcd/mani/ori_feature_{pcd_name}_{i}.pt")

        for i in range(bs):
            newdata = seg_pointcloud(xyz[i], xyz[i], reference_point=output_pos[i], distance=self.feature_point_radius, extra_data={"feature": feature[i]})
            if newdata["xyz"].shape[0] == 0:
                # use the pos point
                output_ori[i] = (feature[i].T * pos_weights[i].detach()).T.sum(dim=0)
            else:
                output_ori[i] = newdata["feature"].mean(dim=0)

        for i in range(3):
            output_ori[:, 3*i:3*(i+1)] /= (torch.norm(output_ori[:, 3*i:3*(i+1)].clone(), dim=1).unsqueeze(1) + 1e-8)
        return output_pos, output_ori

# %%


# %%
model_mani = Model_mani()
model_mani.to("cuda")

# %%
optm = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# %%
scheduler = StepLR(optm, step_size=int(cfg.epoch/5), gamma=0.5)
loss_fn = torch.nn.MSELoss()

# %%
best_test_loss = 1e5

# %%
for epoch in range(cfg.epoch):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epoch}")
    model_mani.train()

    for i, data in enumerate(progress_bar):
        optm.zero_grad()

        with torch.no_grad():
            ref_point = model_mani(
                {"xyz": data["xyz"], "rgb": data["rgb"]}, 
                random_drop=False,
            )

        if cfg.ref_point == "seg_net":
            training_ref_point = ref_point
        elif cfg.ref_point == "gt":
            training_ref_point = data["seg_center"]

        output_pos, output_direction = model_mani(
            {"xyz": data["xyz"], "rgb": data["rgb"]}, 
            reference_point=training_ref_point, 
            distance_threshold=cfg.distance_threshold,
            random_drop=cfg.random_drop,
            train_pos=True,
            draw_pcd=cfg.draw_pcd,
            pcd_name=f"{i}",
            mask_part=cfg.mask_part,
        )

        pos_loss = loss_fn(output_pos, data["seg_center"])
        ori_loss = loss_fn(output_direction, data["axes"])

        if epoch < cfg.pos_warmup_epoch:
            loss = pos_loss
        else:
            loss = pos_loss + 0.1* ori_loss
        loss.backward()
        optm.step()

        with torch.no_grad():
            T1 = torch.zeros([data["axes"].shape[0], 4, 4]).to("cuda")
            T2 = torch.zeros_like(T1).to("cuda")
            T1[:, :3, :3] = data["axes"].reshape(data["axes"].shape[0], 3, 3).transpose(1,2)
            T1[:, :3, 3] = data["seg_center"]
            T1[:, 3, 3] = 1.
            T2[:, :3, :3] = output_direction.reshape(data["axes"].shape[0], 3, 3).transpose(1, 2)
            T2[:, :3, 3] = output_pos
            T2[:, 3, 3] = 1.
            t_loss, r_loss = double_geodesic_distance_between_poses(T1, T2, return_both=True)

        progress_bar.set_postfix(pos_loss=t_loss.item(), ori_loss=r_loss.item())

    model_mani.eval()
    with torch.no_grad():
        test_pos_loss = 0
        test_ori_loss = 0
        for batch_idx, data in enumerate(test_loader):
            output_pos, output_direction = model_mani(
                {"xyz": data["xyz"], "rgb": data["rgb"]}, 
                reference_point=data["seg_center"], 
                distance_threshold=cfg.distance_threshold,
                random_drop=cfg.random_drop,
                train_pos=False,
                draw_pcd=cfg.draw_pcd,
                pcd_name=f"test_{batch_idx}",
            )
            pos_loss = loss_fn(output_pos, data["seg_center"])
            ori_loss = loss_fn(output_direction, data["axes"])

            T1 = torch.zeros([data["axes"].shape[0], 4, 4]).to("cuda")
            T2 = torch.zeros_like(T1).to("cuda")
            T1[:, :3, :3] = data["axes"].reshape(data["axes"].shape[0], 3, 3).transpose(1,2)
            T1[:, :3, 3] = data["seg_center"]
            T1[:, 3, 3] = 1.
            T2[:, :3, :3] = output_direction.reshape(data["axes"].shape[0], 3, 3).transpose(1, 2)
            T2[:, :3, 3] = output_pos
            T2[:, 3, 3] = 1.
            t_loss, r_loss = double_geodesic_distance_between_poses(T1, T2, return_both=True)
            test_pos_loss += t_loss.item()
            test_ori_loss += r_loss.item()

        test_pos_loss /= len(test_loader)
        test_ori_loss /= len(test_loader)
        logging.info(f"Epoch: {epoch},  test pos loss: , {test_pos_loss},  test ori loss: , {test_ori_loss}")
        if test_pos_loss + test_ori_loss < best_test_loss:
            best_test_loss = test_pos_loss + test_ori_loss
            torch.save(model_mani.state_dict(), os.path.join(wd, f"maninet.pth"))
            logging.info("Model saved!")

    scheduler.step()



# %%


# %%


# %%



