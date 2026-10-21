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

from wheel_legged_gym.envs.q1_vmc.q1_vmc_config import (
    Q1VMCCfg,
    Q1VMCCfgPPO,
)


class Q1VMCStairsCfg(Q1VMCCfg):
    """
    Q1 VMC stairs climbing configuration.
    Reuses the Q1 VMC base config and only overrides terrain, commands,
    and reward parameters for stair-climbing training.
    """

    class env(Q1VMCCfg.env):
        episode_length_s = 25

    class init_state(Q1VMCCfg.init_state):
        pos = [0.0, 0.0, 0.40]   # taller spawn so robot clears local terrain

    class terrain(Q1VMCCfg.terrain):
        # ---- curriculum ----
        curriculum = True
        max_init_terrain_level = 0   # start at easiest row
        num_rows = 6                  # 6 difficulty levels
        num_cols = 20

        # Terrain types: [smooth, rough, stairs_up, stairs_down, discrete, stepping_stones]
        # Stairs dominate so the policy spends most capacity on the task that matters.
        terrain_proportions = [0.0, 0.0, 0.5, 0.3, 0.1, 0.1]

        # ---- 10 cm stair target ----
        # step_height = 0.05 + (step_height_max - 0.05) * difficulty
        step_height_max = 0.10   # 10 cm stairs at max difficulty

        # ---- wider forward perception for stair detection ----
        measured_points_x = [
            -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5,
        ]
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

    class commands(Q1VMCCfg.commands):
        curriculum = False   # terrain curriculum only; no command curriculum
        class ranges(Q1VMCCfg.commands.ranges):
            lin_vel_x = [0.3, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            height = [0.15, 0.30]

    class control(Q1VMCCfg.control):
        # VMC gains tuned for Q1 stair climbing.
        feedforward_force = 55.0    # was 40.0
        kp_theta = 60.0             # was 50.0
        kd_theta = 4.0              # was 3.0
        kp_l0 = 1500.0              # was 900.0
        kd_l0 = 30.0                # was 20.0
        # Keys use reliable substrings matching Q1 joint names:
        stiffness = {"thigh": 0.0, "calf": 0.0, "foot": 0.0}   # VMC passive legs
        damping   = {"thigh": 0.0, "calf": 0.0, "foot": 0.6}    # wheel (foot link) PD damping

    class rewards(Q1VMCCfg.rewards):
        """Override scales for stair climbing; inherits non-overridden entries from LeggedRobotCfg.rewards."""
        class scales(Q1VMCCfg.rewards.scales):
            base_height = 1.0
            orientation = -15.0
            lin_vel_z = -1.0
            tracking_lin_vel = 1.5
            tracking_lin_vel_enhance = 1.0
            collision = -1.0
            dof_pos_limits = -1.0

    class domain_rand(Q1VMCCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.3, 1.8]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.85, 1.15]
        randomize_Kp = True
        randomize_Kp_range = [0.85, 1.15]
        randomize_Kd = True
        randomize_Kd_range = [0.85, 1.15]
        push_robots = True
        push_interval_s = 6
        max_push_vel_xy = 1.0


class Q1VMCStairsCfgPPO(Q1VMCCfgPPO):
    class runner(Q1VMCCfgPPO.runner):
        experiment_name = "q1_vmc_stairs"
        run_name = ""
        max_iterations = 10000
