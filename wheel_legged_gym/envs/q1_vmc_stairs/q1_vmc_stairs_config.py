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
        pos = [0.0, 0.0, 0.45]   # taller spawn so robot clears local terrain
        default_joint_angles = {
            "L_thigh_joint": 0.4,
            "L_calf_joint": -0.9,
            "L_foot_joint": 0.0,
            "R_thigh_joint": 0.4,
            "R_calf_joint": -0.9,
            "R_foot_joint": 0.0,
        }

    class terrain(Q1VMCCfg.terrain):
        # ---- curriculum ----
        curriculum = True
        max_init_terrain_level = 0   # start at easiest row
        num_rows = 6                  # 6 difficulty levels
        num_cols = 20

        # Terrain types: [smooth, rough, stairs_up, stairs_down, discrete, stepping_stones]
        # Stairs dominate so the policy spends most capacity on the task that matters.
        terrain_proportions = [0.5, 0.0, 0.25, 0.25, 0.0, 0.0]

        # ---- 10 cm stair target ----
        # step_height = 0.05 + (step_height_max - 0.05) * difficulty
        step_height_max = 0.10   # 10 cm stairs at max difficulty

        # ---- wider forward perception for stair detection ----
        measured_points_x = [
            -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5,
        ]
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

    class commands(Q1VMCCfg.commands):
        curriculum = False  # terrain curriculum remains True; do not change command range
        class ranges(Q1VMCCfg.commands.ranges):
            lin_vel_x = [0.0, 0.6]
            ang_vel_yaw = [-0.3, 0.3]
            height = [0.42, 0.42]  # Q1 body/trunk target height above local terrain

    class control(Q1VMCCfg.control):
        # Keep standing controller consistent with Q1 URDF; start from modest
        # gains for stability before retuning for dynamic 10cm stairs.
        feedforward_force = 34.0
        kp_theta = 35.0
        kd_theta = 2.5
        kp_l0 = 600.0
        kd_l0 = 15.0
        stiffness = {"thigh": 0.0, "calf": 0.0, "foot": 0.0}
        damping = {"thigh": 0.0, "calf": 0.0, "foot": 0.6}

    class rewards(Q1VMCCfg.rewards):
        """Override scales for stair climbing; inherits non-overridden entries from LeggedRobotCfg.rewards."""
        class scales(Q1VMCCfg.rewards.scales):
            tracking_lin_vel = 3.5
            tracking_lin_vel_enhance = 1
            tracking_ang_vel = 2.0

            # Q1 overrides _reward_base_height() with absolute height error.
            # Negative weight gives continuous penalty without exponential vanishing.
            base_height = -3.0
            nominal_state = -0.2
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            orientation = -3.0

            dof_vel = -5e-5
            dof_acc = -2.5e-7
            torques = -0.0001
            action_rate = -0.005
            action_smooth = -0.005

            collision = -2.0
            dof_pos_limits = -1.0

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_single_reward = 1
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = (
            0.97  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.42
        max_contact_force = 1000.0  # forces above this value are penalized

    class domain_rand(Q1VMCCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.3, 1.8]
        randomize_motor_torque = False
        randomize_motor_torque_range = [0.85, 1.15]
        randomize_Kp = False
        randomize_Kp_range = [0.85, 1.15]
        randomize_Kd = False
        randomize_Kd_range = [0.85, 1.15]
        push_robots = False
        push_interval_s = 10
        max_push_vel_xy = 1.0


class Q1VMCStairsCfgPPO(Q1VMCCfgPPO):
    class runner(Q1VMCCfgPPO.runner):
        experiment_name = "q1_vmc_stairs"
        run_name = ""
        max_iterations = 10000
