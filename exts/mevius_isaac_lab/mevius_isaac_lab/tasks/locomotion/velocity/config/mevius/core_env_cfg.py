
from omni.isaac.lab.utils import configclass
from .rough_env_cfg import MeviusRoughEnvCfg
import mevius_isaac_lab

MEVIUS_PATH = mevius_isaac_lab.__path__[0]

@configclass
class MeviusCoreEnvCfg(MeviusRoughEnvCfg):

  def __post_init__(self):
    # post init of parent
    super().__post_init__()

    # override trrain to core field
    self.scene.terrain.terrain_type = "usd"
    self.scene.terrain.terrain_generator = None
    # self.scene.terrain.usd_path = "/home/fkfk21/core_ws/mevius_ws/mevius_isaac_lab/exts/mevius_isaac_lab/data/CoRE/blue-strider-zone.usd"
    # self.scene.terrain.usd_path = f"{MEVIUS_PATH}/data/CoRE/blue-strider-zone.usd"
    self.scene.terrain.usd_path = f"{MEVIUS_PATH}/assets/data/usd/default_environment.usd"
    self.scene.env_spacing = 1.0

    # no terrain curriculum
    self.curriculum.terrain_levels = None





class MeviusCoreEnvCfg_PLAY(MeviusCoreEnvCfg):

  def __post_init__(self):
    # post init of parent
    super().__post_init__()

    # make a smaller scene for play
    self.scene.num_envs = 50
    self.scene.env_spacing = 2.5

    # disable randomization for play
    self.observations.policy.enable_corruption = False

    # remove random pushing event
    self.events.base_external_force_torque = None
    self.events.push_robot = None