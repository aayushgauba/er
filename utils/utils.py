import torch
import torch_scatter
import numpy as np
import dgl
from dgl import backend as F
from dgl.convert import graph as dgl_graph
from networks.se3_transformer.model.fiber import Fiber
from networks.se3_transformer.runtime.utils import to_cuda
from torch_cluster import radius_graph
from scipy.spatial.transform import Rotation

def voxel_filter(pcd, feature, voxel_size: float, coord_reduction: str = "average"):
    """
    Applies voxel filtering to a point cloud.
    
    Args:
        pcd (torch.Tensor): Shape (B, N, 3) – point cloud coordinates.
        feature (torch.Tensor): Shape (B, N, F) – associated features.
        voxel_size (float): Voxel grid size.
        coord_reduction (str): "average" or "center" to reduce voxel coordinates.
        
    Returns:
        coord_vox (torch.Tensor): Voxelized coordinates (for nonzero voxels), shape (V, 3)
        color_vox (torch.Tensor): Voxelized features (for nonzero voxels), shape (V, F)
    """
    device = pcd.device
    print(f"Debug: pcd shape before min: {pcd.shape}")  # e.g., (B, N, 3)

    # Compute the per-batch minimum along the point dimension (dim=1)
    mins = pcd.min(dim=1, keepdim=True).values  # shape: (B, 1, 3)
    print(f"Debug: mins shape after min: {mins.shape}")

    # Compute voxel indices (resulting shape: (B, N, 3))
    vox_idx = torch.div((pcd - mins), voxel_size, rounding_mode='trunc').type(torch.long)
    
    # Convert voxel indices to a NumPy array (shape: (B, N, 3))
    vox_idx_np = vox_idx.cpu().numpy().astype(np.int64)
    
    # Determine the voxel grid shape from the first batch entry.
    # grid_shape: (X, Y, Z) where X = max index in dimension 0 + 1, etc.
    grid_shape = (vox_idx.max(dim=1).values + 1).cpu().numpy().astype(np.int64)  # shape: (B, 3)
    grid_shape = (vox_idx.max(dim=1).values + 1).cpu().numpy().astype(np.int64)  # ideally shape: (B, 3)

    # Check if grid_shape is 2D. If it is, use the first batch entry.
    if grid_shape.ndim == 2:
        grid_shape = tuple(grid_shape[0])
    elif grid_shape.ndim == 1:
        # If it's already 1D, just convert it to a tuple.
        grid_shape = tuple(grid_shape)
    else:
        # Fallback: wrap the scalar in a tuple.
        grid_shape = (grid_shape,)
        print(f"Debug: vox_idx_np shape: {vox_idx_np.shape}, grid_shape: {grid_shape}")

        B, N, _ = vox_idx_np.shape
    # Reshape to 2D: (B*N, 3)
    vox_idx_np_reshaped = vox_idx_np.reshape(-1, 3)
    # Clamp indices so that they lie within valid bounds
    vox_idx_np_reshaped = np.clip(vox_idx_np_reshaped, 0, np.array(grid_shape) - 1)
    # Compute flat (raveled) indices for each point:
    # np.ravel_multi_index expects an array of shape (num_dims, num_points)
    raveled_idx_np = np.ravel_multi_index(vox_idx_np_reshaped.T, grid_shape)
    # Reshape back to (B, N)
    raveled_idx_np = raveled_idx_np.reshape(B, N)
    raveled_idx = torch.tensor(raveled_idx_np, device=device, dtype=torch.long)  # shape: (B, N)

    # Flatten the raveled indices and features for scatter operations
    raveled_idx_flat = raveled_idx.reshape(-1)  # shape: (B*N,)
    num_voxels = int(np.prod(grid_shape))  # total number of voxels in the grid

    # Count points per voxel: scatter over a 1D flat tensor
    src_flat = torch.ones_like(raveled_idx_flat, device=device, dtype=torch.float32)
    n_pts_per_vox = torch_scatter.scatter(src_flat, raveled_idx_flat, dim=0, dim_size=num_voxels)

    # Get indices of nonzero voxels
    nonzero_vox = n_pts_per_vox.nonzero(as_tuple=False)  # shape: (V_nonzero, 1)
    n_pts_per_vox_nonzero = n_pts_per_vox[nonzero_vox].squeeze(-1)  # shape: (V_nonzero,)

    # Scatter features: flatten features to shape (B*N, F)
    F_dim = feature.shape[-1]
    feat_flat = feature.reshape(-1, F_dim)
    color_vox = torch_scatter.scatter(feat_flat, raveled_idx_flat, dim=0, dim_size=num_voxels)
    # Select only nonzero voxels and compute average
    color_vox = color_vox[nonzero_vox].squeeze(1) / n_pts_per_vox_nonzero.unsqueeze(-1)

    # Compute voxelized coordinates
    if coord_reduction == "center":
        # For each nonzero voxel, compute its voxel index coordinate using unravel_index
        coord_vox_np = np.stack(np.unravel_index(nonzero_vox.cpu().numpy().reshape(-1), grid_shape), axis=-1)
        coord_vox = torch.tensor(coord_vox_np, device=device, dtype=vox_idx.dtype)
        # Convert voxel indices to world coordinates: center = index * voxel_size + min + voxel_size/2
        # Here we use the first batch's mins for simplicity.
        coord_vox = coord_vox * voxel_size + mins[0, 0] + (voxel_size / 2)
    elif coord_reduction == "average":
        # Average coordinates per voxel via scatter
        coord_flat = pcd.reshape(-1, 3)
        coord_vox = torch_scatter.scatter(coord_flat, raveled_idx_flat, dim=0, dim_size=num_voxels)
        coord_vox = coord_vox[nonzero_vox].squeeze(1) / n_pts_per_vox_nonzero.unsqueeze(-1)
    else:
        raise ValueError(f"Unknown coordinate reduction method: {coord_reduction}")

    print(f"Debug: Final coord_vox shape: {coord_vox.shape}")  # (V_nonzero, 3)
    return coord_vox, color_vox

def build_graph(xyz, feature, dist_threshold=0.02, self_connection=True, voxelize=False, voxel_size=0.001, fiber_in=Fiber({0: 3})):
    """
    Builds a DGL graph from a point cloud.

    Args:
        xyz (torch.Tensor): Point cloud coordinates.
        feature (torch.Tensor): Feature data.
        dist_threshold (float): Distance threshold for graph edges.
        self_connection (bool): Whether to allow self-connections.
        voxelize (bool): Whether to apply voxelization.
        voxel_size (float): Voxel size for filtering.
        fiber_in (Fiber): Fiber structure.

    Returns:
        batched_graph (DGLGraph): Graph representation.
        node_feats (dict): Node feature dictionary.
        edge_feats (dict): Edge feature dictionary.
        pcds (list): Processed point clouds.
        raw_features (list): Processed feature data.
    """
    bs = xyz.shape[0] if isinstance(xyz, torch.Tensor) else len(xyz)
    batched_graph = []
    pcds = []
    raw_features = []

    max_num_neighbors = 512
    if self_connection:
        max_num_neighbors += 1
    
    for pcd_index in range(bs):
        current_pcd, current_feature = xyz[pcd_index], feature[pcd_index]

        if voxelize:
            current_pcd, current_feature = voxel_filter(current_pcd, current_feature, voxel_size=voxel_size)

        edge_src, edge_dst = radius_graph(
            current_pcd, dist_threshold * 0.999, max_num_neighbors=max_num_neighbors, loop=self_connection
        )

        pcds.append(current_pcd)
        raw_features.append(current_feature)
        g = dgl_graph((edge_src, edge_dst))

        g = to_cuda(g)
        g.ndata["pos"] = F.tensor(current_pcd, dtype=F.data_type_dict["float32"])
        g.ndata["attr"] = F.tensor(current_feature, dtype=F.data_type_dict["float32"])
        g.edata["rel_pos"] = F.tensor(current_pcd[edge_dst] - current_pcd[edge_src])
        g.edata['edge_attr'] = g.edata["rel_pos"].norm(dim=-1)

        batched_graph.append(g)

    batched_graph = dgl.batch(batched_graph)
    batched_graph = to_cuda(batched_graph)

    # Extract node features
    node_feats = {}
    start = 0
    for i, degree in enumerate(fiber_in.degrees):
        feat = batched_graph.ndata['attr'][:, start:start + fiber_in.channels[i] * ((2 * degree) + 1)]
        node_feats[str(degree)] = feat.reshape(feat.shape[0], fiber_in.channels[i], 2 * degree + 1)
        start += fiber_in.channels[i] * ((2 * degree) + 1)
    
    edge_feats = {'0': batched_graph.edata['edge_attr'].unsqueeze(-1).unsqueeze(-1)}

    return batched_graph, node_feats, edge_feats, pcds, raw_features


def modified_gram_schmidt(tensor, to_cuda=False):
    """
    Applies the Modified Gram-Schmidt process for orthonormalization.

    Args:
        tensor (torch.Tensor): Input matrix.
        to_cuda (bool): Move to CUDA.

    Returns:
        torch.Tensor: Orthonormalized matrix.
    """
    rows, cols = tensor.shape
    ortho_tensor = torch.empty_like(tensor)
    if to_cuda:
        ortho_tensor = ortho_tensor.cuda()

    for i in range(cols):
        v = tensor[:, i]
        for j in range(i):
            u = ortho_tensor[:, j]
            v -= torch.dot(v, u) * u
        ortho_tensor[:, i] = v / torch.norm(v)

    return ortho_tensor
