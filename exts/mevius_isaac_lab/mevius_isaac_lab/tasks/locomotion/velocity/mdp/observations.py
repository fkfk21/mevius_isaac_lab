from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

from omni.isaac.lab.assets.articulation.articulation import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor, Imu

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def foot_contact_force_dirs(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the contact forces of the feet """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    force_magnitude = torch.norm(forces, dim=2)
    eps = 1e-10
    force_dir = forces / (force_magnitude.unsqueeze(2) + eps)
    zero_dir = torch.tensor([0.0, 0.0, 0.0], dtype=forces.dtype, device=forces.device)
    # if the force is zero, the direction is 0,0,0
    force_dir[force_magnitude < eps] = zero_dir
    return force_dir.flatten(1)

def foot_contact_forces(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the contact forces of the feet """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_magnitude = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=2)
    return force_magnitude

def foot_friction_coeffs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Return the contact dynamic and static friction coefficients of the feet
    
    Note: This implementation only works when each link has a single shape
          For the case where a link has multiple shapes, 
          see the implementation in `randomize_rigid_body_material` in events.py
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = torch.tensor(np.array(asset_cfg.body_ids, dtype=np.int_), device="cpu")
    env_ids = torch.arange(env.scene.num_envs, device="cpu")
    materials = asset.root_physx_view.get_material_properties()
    return materials[env_ids][:, body_ids, 0:2].flatten(1).to(env.device)

def imu_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor linear velocity w.r.t. environment origin expressed in the sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The linear velocity (m/s) in the sensor frame. Shape is (num_envs, 3).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the linear velocity
    return asset.data.lin_vel_b