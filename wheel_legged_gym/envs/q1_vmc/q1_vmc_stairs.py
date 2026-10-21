# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Q1 VMC stairs environment — reuses LeggedRobotQ1VMC with stairs terrain config.

from wheel_legged_gym.envs.q1_vmc.q1_vmc import LeggedRobotQ1VMC
from .q1_vmc_stairs_config import Q1VMCStairsCfg

# No code changes needed: LeggedRobotQ1VMC + Q1VMCStairsCfg together give
# the correct VMC joint mappings with stairs terrain and curriculum.
