# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT


from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from ....se3_transformer.model.fiber import Fiber
from contextlib import contextmanager

@contextmanager
def nvtx_range(label):
    # Dummy NVTX range: does nothing, for CPU-only environments.
    yield

@torch.jit.script
def clamped_norm(x, clamp: float):
    return x.norm(p=2, dim=-1, keepdim=True).clamp(min=clamp)

@torch.jit.script
def rescale(x, norm, new_norm):
    return x / norm * new_norm


class NormSE3(nn.Module):
    """
    Norm-based SE(3)-equivariant nonlinearity.

                 ┌──> feature_norm ──> LayerNorm() ──> ReLU() ──┐
    feature_in ──┤                                              * ──> feature_out
                 └──> feature_phase ────────────────────────────┘
    """

    NORM_CLAMP = 2 ** -24  # Minimum positive subnormal for FP16

    def __init__(self, fiber: Fiber, nonlinearity: nn.Module = nn.ReLU()):
        super().__init__()
        self.fiber = fiber
        self.nonlinearity = nonlinearity

        if len(set(fiber.channels)) == 1:
            # Fuse all the layer normalizations into a group normalization
            self.group_norm = nn.GroupNorm(num_groups=len(fiber.degrees), num_channels=sum(fiber.channels))
        else:
            # Use multiple layer normalizations
            self.layer_norms = nn.ModuleDict({
                str(degree): nn.LayerNorm(channels)
                for degree, channels in fiber
            })

    def forward(self, features: Dict[str, Tensor], *args, **kwargs) -> Dict[str, Tensor]:
        with nvtx_range('NormSE3'):
            output = {}
            if hasattr(self, 'group_norm'):
                # Compute per-degree norms of features
                norms = [clamped_norm(features[str(d)], self.NORM_CLAMP)
                         for d in self.fiber.degrees]
                fused_norms = torch.cat(norms, dim=-2)

                # Transform the norms only
                new_norms = self.nonlinearity(self.group_norm(fused_norms.squeeze(-1))).unsqueeze(-1)
                new_norms = torch.chunk(new_norms, chunks=len(self.fiber.degrees), dim=-2)

                # Scale features to the new norms
                for norm, new_norm, d in zip(norms, new_norms, self.fiber.degrees):
                    output[str(d)] = rescale(features[str(d)], norm, new_norm)
            else:
                for degree, feat in features.items():
                    norm = clamped_norm(feat, self.NORM_CLAMP)
                    new_norm = self.nonlinearity(self.layer_norms[degree](norm.squeeze(-1)).unsqueeze(-1))
                    output[degree] = rescale(new_norm, feat, norm)

            return output
