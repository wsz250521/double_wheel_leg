# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Q1-specific VMC adaptation.  The upstream wheel-legged VMC remains unchanged.

"""VMC for Q1's two serial hip/knee legs and two wheels.

URDF (trunk frame): both hip and knee axes are +Y, on both sides.
Positive knee/hip angle rotates the leg toward negative trunk X.
The Q1 leg endpoints are wheel-axle centers (not ground-contact points).

Body-frame leg endpoint (x forward, z_down toward ground):
    x = hip_x - l1*sin(q_hip) - l2*sin(q_hip+q_knee)
    z = -hip_z + l1*cos(q_hip) + l2*cos(q_hip+q_knee)
    L = sqrt(x*x + z*z),  theta = atan2(x,z)

Joint torques use the analytical Jacobian transpose of (L, theta).
The Q1 controller, not the original WL controller, defines all joint signs.
"""

import torch

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC
from .q1_vmc_config import Q1VMCCfg


class LeggedRobotQ1VMC(LeggedRobotVMC):
    _EXPECTED_DOF_NAMES = (
        "L_thigh_joint", "L_calf_joint", "L_foot_joint",
        "R_thigh_joint", "R_calf_joint", "R_foot_joint",
    )

    def __init__(self, cfg: Q1VMCCfg, sim_params, physics_engine, sim_device, headless):
        # Normal inherited init runs LeggedRobotVMC._init_buffers through
        # virtual method dispatch, which initializes all VMC gain tensors.
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        actual = tuple(self.dof_names)
        if actual != self._EXPECTED_DOF_NAMES:
            raise ValueError(
                "Q1 VMC joint order differs from the URDF assumption. "
                f"Expected {self._EXPECTED_DOF_NAMES}, actual {actual}. "
                "Adjust joint indexing before applying torques."
            )
        self.leg_post_physics_step()

    def _q1_kinematics(self, hip, knee):
        """Returns L, theta and analytical d(L,theta)/d(hip,knee).

        Args: hip, knee: [num_envs,2] both legs, using actual URDF DOF signs.
        Returns: L,theta [N,2]; J_L,J_theta [N,2,2] with last axis hip,knee.
        """
        q12 = hip + knee
        l1 = self.cfg.asset.l1
        l2 = self.cfg.asset.l2
        x = (self.cfg.asset.hip_x
             - l1 * torch.sin(hip) - l2 * torch.sin(q12))
        z = (-self.cfg.asset.hip_z
             + l1 * torch.cos(hip) + l2 * torch.cos(q12))
        length_sq = torch.clamp(x.square() + z.square(), min=1e-8)
        length = torch.sqrt(length_sq)
        theta = torch.atan2(x, z)  # zero is vertical; positive means axle in +X

        dx_dhip = -l1 * torch.cos(hip) - l2 * torch.cos(q12)
        dx_dknee = -l2 * torch.cos(q12)
        dz_dhip = -l1 * torch.sin(hip) - l2 * torch.sin(q12)
        dz_dknee = -l2 * torch.sin(q12)

        dlength_dhip = (x * dx_dhip + z * dz_dhip) / length
        dlength_dknee = (x * dx_dknee + z * dz_dknee) / length
        dtheta_dhip = (z * dx_dhip - x * dz_dhip) / length_sq
        dtheta_dknee = (z * dx_dknee - x * dz_dknee) / length_sq

        jac_length = torch.stack((dlength_dhip, dlength_dknee), dim=-1)
        jac_theta = torch.stack((dtheta_dhip, dtheta_dknee), dim=-1)
        return length, theta, jac_length, jac_theta

    def forward_kinematics(self, theta1, theta2):
        """Q1-only replacement for WL forward kinematics.

        Inputs are Q1 *actual* hip and knee joint angles, not WL transformed
        angles (no right-leg negation and no knee +pi/2 offset).
        """
        length, theta, _, _ = self._q1_kinematics(theta1, theta2)
        return length, theta

    def leg_post_physics_step(self):
        # Actual Q1 URDF order: L_hip,L_knee,L_wheel,R_hip,R_knee,R_wheel.
        # Both sides have +Y joint axes, so right angles MUST NOT be negated.
        self.theta1 = self.dof_pos[:, [0, 3]]
        self.theta2 = self.dof_pos[:, [1, 4]]
        hip_vel = self.dof_vel[:, [0, 3]]
        knee_vel = self.dof_vel[:, [1, 4]]

        self.L0, self.theta0, self._q1_jac_length, self._q1_jac_theta = (
            self._q1_kinematics(self.theta1, self.theta2)
        )
        qdot = torch.stack((hip_vel, knee_vel), dim=-1)
        self.L0_dot = torch.sum(self._q1_jac_length * qdot, dim=-1)
        self.theta0_dot = torch.sum(self._q1_jac_theta * qdot, dim=-1)
    
    def _reward_stair_gait(self):
        """
        台阶触发式抬腿步态奖励。

        1. 前方存在上升台阶时才激活。
        2. 鼓励腿部缩短或摆动，从而抬高轮子。
        3. 鼓励左右腿产生差异化动作。
        4. 鼓励抬腿期间保持前进和机身稳定。

        返回: [num_envs]
        """

        # ---------- 1. 平地不启用 ----------
        if self.cfg.terrain.mesh_type not in ["trimesh", "heightfield"]:
            return torch.zeros(self.num_envs, device=self.device)

        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)

        # ---------- 2. 检测前方台阶 ----------

        # 地形采样点在机器人局部坐标系中的 X 坐标
        points_x = self.height_points[0, :, 0]

        # 前方 0.1~0.5m 范围内的地形
        front_mask = (points_x >= 0.1) & (points_x <= 0.5)

        # 后方 -0.3~-0.1m 作为当前地面参考
        back_mask = (points_x >= -0.3) & (points_x <= -0.1)

        # 前方最高地形
        front_height = torch.max(
            self.measured_heights[:, front_mask],
            dim=1
        ).values

        # 后方地形参考高度
        back_height = torch.median(
            self.measured_heights[:, back_mask],
            dim=1
        ).values

        # 前后地形高度差
        step_height = torch.clamp(
            front_height - back_height,
            min=0.0
        )

        # 高度差超过 2cm 开始激活
        # 5cm 及以上时完全激活
        step_gate = torch.clamp(
            (step_height - 0.02) / 0.03,
            min=0.0,
            max=1.0
        )

        # ---------- 3. 计算腿部抬升程度 ----------

        # 默认站立腿长
        nominal_l0 = self.cfg.control.l0_offset

        # 当前轮轴相对机身的竖直距离
        # theta0 接近 0 时，腿基本竖直
        vertical_length = self.L0 * torch.cos(self.theta0)

        # 相对于默认姿态的轮轴抬升量
        # 这是相对机身的运动学抬升量，不是实际离地高度
        lift_height = torch.clamp(
            nominal_l0 - vertical_length,
            min=0.0
        )

        # 抬升 4cm 左右可获得较高奖励
        lift_score = torch.clamp(
            lift_height / 0.04,
            min=0.0,
            max=1.0
        )

        left_lift = lift_score[:, 0]
        right_lift = lift_score[:, 1]

        # ---------- 4. 步态协调奖励 ----------

        # 鼓励至少有一条腿主动抬升
        active_lift = torch.maximum(
            left_lift,
            right_lift
        )

        # 鼓励左右腿动作产生差异
        gait_difference = torch.abs(
            left_lift - right_lift
        )

        # 允许同步抬腿，但更倾向于差异化动作
        gait_reward = (
            0.7 * active_lift
            + 0.3 * gait_difference
        )

        # ---------- 5. 前进奖励门控 ----------

        # 鼓励抬腿时继续向前
        forward_factor = torch.clamp(
            self.base_lin_vel[:, 0] / 0.3,
            min=0.0,
            max=1.0
        )

        # 保留基础奖励，避免撞上台阶停住后奖励完全消失
        forward_factor = 0.3 + 0.7 * forward_factor

        # ---------- 6. 姿态稳定门控 ----------

        upright_factor = torch.clamp(
            (-self.projected_gravity[:, 2] - 0.6) / 0.4,
            min=0.0,
            max=1.0
        )

        # ---------- 7. 最终奖励 ----------

        reward = (
            step_gate
            * gait_reward
            * forward_factor
            * upright_factor
        )

        return reward

    def VMC(self, F, T):
        """Map Q1 virtual axial-force and angle-torque into URDF joint torques.

        F > 0 pushes toward longer virtual-leg length; the Jacobian transpose
        handles joint signs consistently for BOTH Q1 legs.
        """
        joint_torque = (self._q1_jac_length * F.unsqueeze(-1)
                        + self._q1_jac_theta * T.unsqueeze(-1))
        return joint_torque[..., 0], joint_torque[..., 1]

    def _compute_torques(self, actions):
        """Q1 actions: [L_theta,L_length,L_wheel,R_theta,R_length,R_wheel]."""
        # Limit commanded offsets while keeping learned policy/action API intact.
        a = torch.clamp(actions, -1.0, 1.0)
        theta_ref = a[:, [0, 3]] * self.cfg.control.action_scale_theta
        length_ref = (
            a[:, [1, 4]] * self.cfg.control.action_scale_l0
            + self.cfg.control.l0_offset
        )
        wheel_vel_ref = a[:, [2, 5]] * self.cfg.control.action_scale_vel

        self.torque_leg = (
            self.theta_kp * (theta_ref - self.theta0)
            - self.theta_kd * self.theta0_dot
        )
        self.force_leg = (
            self.l0_kp * (length_ref - self.L0)
            - self.l0_kd * self.L0_dot
        )
        self.torque_wheel = (
            self.d_gains[:, [2, 5]] * (wheel_vel_ref - self.dof_vel[:, [2, 5]])
        )
        hip_torque, knee_torque = self.VMC(
            self.force_leg + self.cfg.control.feedforward_force, self.torque_leg
        )

        # Unlike WL, Q1 has the SAME +Y axis sign on both sides.
        torques = torch.stack(
            (hip_torque[:, 0], knee_torque[:, 0], self.torque_wheel[:, 0],
             hip_torque[:, 1], knee_torque[:, 1], self.torque_wheel[:, 1]),
            dim=-1,
        )
        return torch.clamp(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def _reward_base_height(self):
        """Smooth NON-SATURATING local base-height penalty for Q1.

        On stairs, use samples below/very near the trunk, not the average over
        1.5m of *look-ahead* terrain.  Reward scale must be negative.
        """
        local_terrain = None
        samples = self.measured_heights
        if torch.is_tensor(samples) and samples.ndim == 2:
            xs = self.cfg.terrain.measured_points_x
            ys = self.cfg.terrain.measured_points_y
            near_x = [i for i, v in enumerate(xs) if abs(v) <= 0.15]
            near_y = [i for i, v in enumerate(ys) if abs(v) <= 0.20]
            indices = [i * len(ys) + j for i in near_x for j in near_y]
            if indices and samples.shape[1] == len(xs) * len(ys):
                local_terrain = samples[:, indices].mean(dim=1)
        if local_terrain is None:
            # Fallback for plane or different terrain sampling configuration.
            local_height = self.base_height
        else:
            local_height = self.root_states[:, 2] - local_terrain
        return torch.abs(local_height - self.commands[:, 2])

    def _reward_nominal_state(self):
        """Penalize asymmetric/splayed legs, with a mild posture regularizer."""
        diff = self.theta0[:, 0] - self.theta0[:, 1]
        return diff.square() + 0.2 * torch.sum(self.theta0.square(), dim=1)

    def reset_idx(self, env_ids):
        # Q1 state is refreshed after the parent reset to avoid stale VMC values.
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            self.leg_post_physics_step()
