# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""
Q1 robot VMC environment.

Key differences from wheel-legged VMC:
- 6 DOFs without abduction joints: [L_thigh, L_calf, L_foot, R_thigh, R_calf, R_foot]
- Joint indices for VMC state extraction:
    theta1: L_thigh (idx 0), R_thigh (idx 3)
    theta2: L_calf  (idx 1), R_calf  (idx 4)
  (wheel-legged uses idx 1/4 for theta1/theta2 because it has abduction at idx 0/3)
- Wheel joints are indices 2 and 5 (same as wheel-legged).

The VMC math (forward_kinematics, VMC block) is unchanged since both robots
are planar 2-link mechanisms in the sagittal plane.
"""

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC
from .q1_vmc_config import Q1VMCCfg
import torch


class LeggedRobotQ1VMC(LeggedRobotVMC):
    """
    Q1 robot with VMC-based controller.
    Inherits all training infrastructure (terrain, rewards, domain rand, PPO runner)
    from LeggedRobotVMC; only overrides the VMC-specific joint mappings.
    """

    def __init__(
        self, cfg: Q1VMCCfg, sim_params, physics_engine, sim_device, headless
    ):
        self.cfg = cfg
        # Delegating to LeggedRobot (grand-parent) which calls create_sim(),
        # _init_buffers(), etc., then LeggedRobotVMC adds VMC buffers.
        from wheel_legged_gym.envs.base.legged_robot import LeggedRobot
        LeggedRobot.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)
        # Prime VMC state from the initial DOF positions after environment creation.
        self.leg_post_physics_step()

    def leg_post_physics_step(self):
        """
        Extract VMC state (theta1, theta2, L0, theta0) from Q1 joint positions.
        Q1 DOF order: [L_thigh=0, L_calf=1, L_foot=2, R_thigh=3, R_calf=4, R_foot=5]
        This differs from wheel-legged: [lf0=0, lf1=1, l_wheel=2, rf0=3, rf1=4, r_wheel=5]
        For Q1, thigh is at idx 0/3 (no abduction at idx 0/3); calf at idx 1/4 (same as wheel-legged).
        """
        # theta1: thigh angles.  Left is positive, right is negated (URDF axis sign).
        self.theta1 = torch.cat(
            (self.dof_pos[:, 0].unsqueeze(1), -self.dof_pos[:, 3].unsqueeze(1)), dim=1
        )
        # theta2: calf angles (same indexing as wheel-legged; pi/2 offset converts
        # to VMC convention where theta2=0 means leg is straight).
        self.theta2 = torch.cat(
            (
                (self.dof_pos[:, 1] + self.pi / 2).unsqueeze(1),
                (-self.dof_pos[:, 4] + self.pi / 2).unsqueeze(1),
            ),
            dim=1,
        )
        theta1_dot = torch.cat(
            (self.dof_vel[:, 0].unsqueeze(1), -self.dof_vel[:, 3].unsqueeze(1)), dim=1
        )
        theta2_dot = torch.cat(
            (self.dof_vel[:, 1].unsqueeze(1), -self.dof_vel[:, 4].unsqueeze(1)), dim=1
        )

        self.L0, self.theta0 = self.forward_kinematics(self.theta1, self.theta2)

        # Numerical differentiation for L0_dot and theta0_dot.
        dt = 0.001
        L0_temp, theta0_temp = self.forward_kinematics(
            self.theta1 + theta1_dot * dt, self.theta2 + theta2_dot * dt
        )
        self.L0_dot = (L0_temp - self.L0) / dt
        self.theta0_dot = (theta0_temp - self.theta0) / dt

    def reset_idx(self, env_ids):
        """
        Override reset_idx to also update VMC state after DOF reset,
        so the first post-reset observation contains correct VMC state.
        """
        from wheel_legged_gym.envs.base.legged_robot import LeggedRobot
        LeggedRobot.reset_idx(self, env_ids)
        if len(env_ids) > 0:
            self.leg_post_physics_step()
