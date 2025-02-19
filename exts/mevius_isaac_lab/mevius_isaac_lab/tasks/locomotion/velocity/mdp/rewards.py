from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets.articulation.articulation import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float
) -> torch.Tensor:
    """Reward the robot for maintaining a specific height above the ground using L2-kernel.

    This function rewards the agent for maintaining a specific height above the ground. The reward is computed
    as the L2-norm of the difference between the current height and the target height.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2].unsqueeze(-1)
    # print(base_height[1:5, 0])
    reward = torch.norm(base_height - target_height, dim=1)
    return reward


def feet_stumble(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold_ratio: float
) -> torch.Tensor:
    """Penalize feet hitting the vertical surface. """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # decompose the contact forces
    xy_forces = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids, :2], dim=3)  # (N, SENSOR_NUM)
    z_forces = torch.abs(net_contact_forces[:, :, sensor_cfg.body_ids, 2])  # (N, SENSOR_NUM)
    return torch.sum(torch.any(xy_forces > threshold_ratio*z_forces, dim=2), dim=1)


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_lin_vel_threshold: float = 0.1,
    cmd_ang_vel_threshold: float = 0.1,
) -> torch.Tensor:
    # Penalize motion at nearly zero velocity.
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_lin_vel = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    cmd_ang_vel = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    return torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1) * \
                ((cmd_lin_vel < cmd_lin_vel_threshold) & (cmd_ang_vel < cmd_ang_vel_threshold))
