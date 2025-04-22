import math
from omni.isaac.lab.managers.scene_entity_cfg import SceneEntityCfg
from omni.isaac.lab.utils import configclass
from .rough_env_cfg import MeviusRoughEnvCfg
from mevius_isaac_lab.tasks.locomotion.velocity.velocity_env_cfg import (
    TerminationsCfg
)
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from mevius_isaac_lab.tasks.locomotion.velocity import mdp
import mevius_isaac_lab

MEVIUS_PATH = mevius_isaac_lab.__path__[0]


@configclass
class MeviusCoREEnvCfg(MeviusRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # terrain
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.10)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.09)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.02, 0.20)
        self.scene.terrain.terrain_generator.sub_terrains["steps"] = mdp.terrains.MeshConsecutiveStepsTerrainCfg(
            step_height_range=(0.02, 0.17),
            step_width_range=(0.25, 0.45),
            step_margin_range=(0.35, 0.45),
            border_width=0.5,
            platform_width=1.0,
        )

        # terrain proportions
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion       = 0.1
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion   = 0.1
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion         = 0.2
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion                = 0.2
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion     = 0.1
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.1
        self.scene.terrain.terrain_generator.sub_terrains["steps"].proportion                = 0.2

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class MeviusCoREEnvCfg_PLAY(MeviusCoREEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 100
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class CoREFieldEnvTerminationsCfg(TerminationsCfg):
    arrive_goal = DoneTerm(
        func=mdp.arrive_goal,
        time_out=True,
        params={"goal_x": 6.6, "asset_cfg": SceneEntityCfg("robot", body_names="base")}
    )

    def __post_init__(self):
        super().__post_init__()


@configclass
class MeviusCoREFieldEnvCfg_PLAY(MeviusCoREEnvCfg):
    terminations = CoREFieldEnvTerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 9
        self.scene.env_spacing = 8

        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.9)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0, 0)}

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # override trrain to core field
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.usd_path = f"{MEVIUS_PATH}/assets/data/CoRE/strider-zone.usd"

        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None