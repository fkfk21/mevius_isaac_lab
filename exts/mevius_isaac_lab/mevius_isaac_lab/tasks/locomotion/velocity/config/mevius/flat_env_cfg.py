from omni.isaac.lab.utils import configclass

from .rough_env_cfg import MeviusRoughEnvCfg


@configclass
class MeviusFlatEnvCfg(MeviusRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # action scale
        self.actions.joint_pos.scale = 0.5

        # command scale
        self.commands.base_velocity.heading_command = False
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.05)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.05)


        # event params
        self.events.physics_material.params["static_friction_range"] = (0.7, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.0)

        # reward scales
        self.rewards.base_height_l2.weight = -0.01
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.dof_acc_l2.weight = -2.0e-9
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.dof_vel_l2.weight = -2.5e-9
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 0.9 
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.termination.weight = -0.0
        self.rewards.stand_still.weight = -1.0 # -10.0

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class MeviusFlatEnvCfg_PLAY(MeviusFlatEnvCfg):
    def __post_init__(self) -> None:
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
