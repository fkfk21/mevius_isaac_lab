
from omni.isaac.lab.utils import configclass
from .rough_env_cfg import MeviusRoughEnvCfg
import mevius_isaac_lab

MEVIUS_PATH = mevius_isaac_lab.__path__[0]

@configclass
class MeviusCoREEnvCfg_PLAY(MeviusRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot.init_state.pos = (0.0, 0.0, 0.33)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # make a smaller scene for play
        self.scene.num_envs = 6
        self.scene.env_spacing = 8

        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 3.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.7, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-1.57, -1.57)}

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