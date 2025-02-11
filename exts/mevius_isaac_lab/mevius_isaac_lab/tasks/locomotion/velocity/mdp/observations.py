from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets.articulation.articulation import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv



def foot_contact_forces(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the contact forces of the feet """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.net_forces_w


def foot_friction_coeffs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the contact dynamic and static friction coefficients of the feet """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    env_ids = torch.arange(env.scene.num_envs, device="cpu")
    materials = asset.root_physx_view.get_material_properties()
    print("env_ids", env_ids)
    print("body_ids", body_ids)
    print("materials", materials)
    return materials[env_ids, body_ids]
