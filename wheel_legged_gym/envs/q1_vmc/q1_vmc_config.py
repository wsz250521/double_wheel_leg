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

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc_config import (
    WheelLeggedVMCCfg,
    WheelLeggedVMCCfgPPO,
)


class Q1VMCCfg(WheelLeggedVMCCfg):
    """
    Q1 robot VMC configuration.
    Q1 has 6 DOFs (no abduction joints):
      [L_thigh, L_calf, L_foot(wheel), R_thigh, R_calf, R_foot(wheel)]
    Inherits RL/training settings from WheelLeggedVMCCfg, while q1_vmc.py
    replaces the WL kinematics and VMC joint-torque mapping.
    """

    class init_state(WheelLeggedVMCCfg.init_state):
        pos = [0.0, 0.0, 0.45]  # Q1 default resting height ~0.426 m
        default_joint_angles = {
            "L_thigh_joint": 0.4,
            "L_calf_joint": -0.9,
            "L_foot_joint": 0.0,
            "R_thigh_joint": 0.4,
            "R_calf_joint": -0.9,
            "R_foot_joint": 0.0,
        }

    class commands(WheelLeggedVMCCfg.commands):
        class ranges(WheelLeggedVMCCfg.commands.ranges):
            # In this repo, commands[:,2] is the *body height* reference.
            height = [0.42, 0.42]

    class control(WheelLeggedVMCCfg.control):
        # Q1 wheel-axle distance from trunk at default angles is ~0.35137m.
        l0_offset = 0.3514
        # Keep actions around physically reasonable positions near standing.
        action_scale_theta = 0.20    # rad, axle fore/aft relative to trunk
        action_scale_l0 = 0.04       # m; length target [0.3114, 0.3914]
        action_scale_vel = 8.0       # rad/s
        # Q1 model is ~6.87kg, so static load is ~33.7N per leg.
        feedforward_force = 34.0
        kp_theta = 35.0
        kd_theta = 2.5
        kp_l0 = 600.0
        kd_l0 = 15.0
        stiffness = {"thigh": 0.0, "calf": 0.0, "foot": 0.0}
        damping = {"thigh": 0.0, "calf": 0.0, "foot": 0.6}

    class asset(WheelLeggedVMCCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/q1_description/urdf/q1.urdf"
        name = "Q1"
        # Q1 trunk->hip origin per URDF: (+Y left; -Y right).  2D
        # sagittal FK uses ONLY X and Z.  The old WL offset=0.112 was wrong.
        hip_x = -0.025225
        hip_z = 0.0084361
        offset = 0.0     # unused by Q1's overridden FK
        l1 = 0.2         # hip -> knee
        l2 = 0.2         # knee -> wheel axle
        penalize_contacts_on = ["trunk"]
        terminate_after_contacts_on = ["trunk"]
        self_collisions = 1   # disable self-collisions
        flip_visual_attachments = False


class Q1VMCCfgPPO(WheelLeggedVMCCfgPPO):
    class runner(WheelLeggedVMCCfgPPO.runner):
        experiment_name = "q1_vmc"
        run_name = ""
        max_iterations = 5000
