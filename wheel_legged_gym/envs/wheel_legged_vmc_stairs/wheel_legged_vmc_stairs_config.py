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
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Stair-climbing specialization built on top of `wheel_legged_vmc`.
# Goals:
#   * Train a policy that can climb 10 cm stairs.
#   * Use terrain curriculum learning (start easy, ramp up).
#   * Reuse the `LeggedRobotVMC` class (no custom .py needed).

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc_config import (
    WheelLeggedVMCCfg,
    WheelLeggedVMCCfgPPO,
)


class WheelLeggedVMCStairsCfg(WheelLeggedVMCCfg):

    class env(WheelLeggedVMCCfg.env):
        # Slightly longer episodes: stair sequences need more time to complete.
        episode_length_s = 25

    class init_state(WheelLeggedVMCCfg.init_state):
        # Lift the spawn height a bit so the robot starts above the local
        # terrain when on the easy rows of the curriculum.
        pos = [0.0, 0.0, 0.30]

    class terrain(WheelLeggedVMCCfg.terrain):
        # ---- curriculum ----
        curriculum = True
        # Start every robot at the easiest row. As policies succeed they
        # progress upward automatically (see `_update_terrain_curriculum`).
        max_init_terrain_level = 0
        # Six difficulty rows gives fine-grained progression around 10 cm.
        num_rows = 6
        num_cols = 20  # keep the original column layout

        # Terrain types: [smooth slope, rough slope, stairs up, stairs down,
        # discrete, stepping stones]. Stairs dominate so the policy spends
        # most of its capacity on the task that matters. The values are
        # chosen to sum to exactly 1.0 in IEEE-754 (no float drift).
        terrain_proportions = [0.0, 0.0, 0.5, 0.3, 0.1, 0.1]

        # ---- final target: 10 cm stairs ----
        # The terrain generator uses step_height = 0.05 + (step_height_max - 0.05) * difficulty
        # so this caps the tallest stair the curriculum ever produces.
        step_height_max = 0.10

        # ---- wider forward perception ----
        # Default samples only [-0.5, 0.5] m ahead. For stairs the policy
        # needs to *see* the step in time to lift the leg, so extend the
        # forward range.
        measured_points_x = [
            -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5,
        ]
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

    class commands(WheelLeggedVMCCfg.commands):
        # No command curriculum for stairs: we want the policy to push
        # forward at a useful speed every episode, not "earn" the right to
        # move faster. The terrain curriculum is the only curriculum that
        # runs here.
        curriculum = False
        # Lower the max forward speed a touch — stairs reward control more
        # than raw speed, and 1.0 m/s is plenty for 10 cm steps.
        class ranges(WheelLeggedVMCCfg.commands.ranges):
            lin_vel_x = [0.3, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            # Bias the commanded body height a bit higher so the policy
            # learns to keep its center of mass over the step edge.
            height = [0.15, 0.25]

    class control(WheelLeggedVMCCfg.control):
        # ---- VMC gains tuned for stair climbing ----
        # Stairs need a stronger feedforward push (the wheel cannot power
        # itself onto the step), a stiffer virtual leg length spring, and
        # a bit more damping to absorb landing impact.
        feedforward_force = 55.0   # was 40.0
        kp_theta            = 60.0  # was 50.0
        kd_theta            = 4.0   # was 3.0
        kp_l0               = 1500.0  # was 900.0
        kd_l0               = 30.0    # was 20.0

        # Keep the open-chain joints passive (the VMC layer drives them).
        stiffness = {"f0": 0.0, "f1": 0.0, "wheel": 0.0}
        damping   = {"f0": 0.0, "f1": 0.0, "wheel": 0.6}  # slightly more wheel damping

    class rewards(WheelLeggedVMCCfg.rewards):
        # Bump rewards that fight stair-induced pitch / height variation.
        class scales(WheelLeggedVMCCfg.rewards.scales):
            # Keep the body upright above the step.
            orientation = -15.0
            # Tolerate vertical bouncing more, otherwise the policy freezes
            # on every landing impact.
            lin_vel_z = -1.0
            # Encourage the policy to actually climb (any forward motion is
            # already covered by `tracking_lin_vel`, but a small bonus on
            # net forward progress speeds up curriculum advancement).
            tracking_lin_vel = 1.5
            tracking_lin_vel_enhance = 1.0

    class domain_rand(WheelLeggedVMCCfg.domain_rand):
        # Stairs are contact-heavy: more variation on friction and torque
        # helps the policy survive sim-to-real.
        randomize_friction = True
        friction_range = [0.3, 1.8]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.85, 1.15]
        randomize_Kp = True
        randomize_Kp_range = [0.85, 1.15]
        randomize_Kd = True
        randomize_Kd_range = [0.85, 1.15]
        # Pushes help explore recovery behaviors on top of stairs.
        push_robots = True
        push_interval_s = 6
        max_push_vel_xy = 1.0


class WheelLeggedVMCStairsCfgPPO(WheelLeggedVMCCfgPPO):

    class runner(WheelLeggedVMCCfgPPO.runner):
        # logging
        experiment_name = "wheel_legged_vmc_stairs"
        run_name = ""
        # Stair curriculum needs more iterations to walk through 6 rows.
        max_iterations = 5000
