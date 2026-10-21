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
    Inherits from WheelLeggedVMCCfg so the VMC controller (theta/l0 VMC, wheel PD)
    is reused; only the asset (URDF), init_state, and action scales differ.
    """

    class init_state(WheelLeggedVMCCfg.init_state):
        pos = [0.0, 0.0, 0.35]  # Q1 body height
        default_joint_angles = {
            "L_thigh_joint": 0.4,
            "L_calf_joint": -0.9,
            "L_foot_joint": 0.0,
            "R_thigh_joint": 0.4,
            "R_calf_joint": -0.9,
            "R_foot_joint": 0.0,
        }

    class control(WheelLeggedVMCCfg.control):
        # VMC action scales tuned for Q1's joint limits (theta: ±1.57, calf: -2.6..0).
        action_scale_theta = 0.5    # theta target range: [-0.785, 0.785] rad
        action_scale_l0 = 0.1       # l0 range: [l0_offset - 0.1, l0_offset + 0.1] m
        action_scale_vel = 8.0      # wheel velocity ref range: [-8, 8] rad/s
        # Keys are matched as substrings in joint names (key in joint_name).
        # Must be substrings of the actual joint names for the matching to succeed.
        stiffness = {"thigh": 0.0, "calf": 0.0, "foot": 0.0}   # VMC passive legs
        damping   = {"thigh": 0.0, "calf": 0.0, "foot": 0.6}    # wheel (foot link) PD damping

    class asset(WheelLeggedVMCCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/q1_description/urdf/q1.urdf"
        name = "Q1"
        # Q1 kinematics: offset from trunk origin to hip joint center.
        offset = 0.112    # sqrt(0.025^2 + 0.109^2) ≈ 0.112 m
        l1 = 0.2          # thigh link: hip → knee ≈ 0.2 m
        l2 = 0.2          # calf link: knee → footmotor ≈ 0.2 m
        penalize_contacts_on = ["trunk"]
        terminate_after_contacts_on = ["trunk"]
        self_collisions = 1   # disable self-collisions
        flip_visual_attachments = False


class Q1VMCCfgPPO(WheelLeggedVMCCfgPPO):
    class runner(WheelLeggedVMCCfgPPO.runner):
        experiment_name = "q1_vmc"
        run_name = ""
        max_iterations = 5000
