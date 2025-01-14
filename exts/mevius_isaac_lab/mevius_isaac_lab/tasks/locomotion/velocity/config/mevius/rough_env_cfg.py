import math
from omni.isaac.lab.managers.scene_entity_cfg import SceneEntityCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from mevius_isaac_lab.tasks.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    RewardsCfg,
    ObservationsCfg,
)
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from mevius_isaac_lab.tasks.locomotion.velocity import mdp
from mevius_isaac_lab.assets.mevius import MEVIUS_CFG


@configclass
class MeviusRewardsCfg(RewardsCfg):
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "target_height": 0.3,
        }
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold_ratio": 5
        }
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=0.00,
        params={
            "command_name": "base_velocity",
        }
    )
    gait = RewTerm(
        func=spot_mdp.GaitReward,
        weight=0.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.1,
            "synced_feet_pair_names": (("FL_foot", "BR_foot"), ("FR_foot", "BL_foot")), 
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        }
    )
    


@configclass
class MeviusSceneCfg(MySceneCfg):
    height_scanner = None


@configclass
class MeviusObservationsCfg(ObservationsCfg):

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        height_scan = None

    policy: PolicyCfg = PolicyCfg()


@configclass
class MeviusRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: MeviusSceneCfg = MeviusSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: MeviusObservationsCfg = MeviusObservationsCfg()
    rewards: MeviusRewardsCfg = MeviusRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = MEVIUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # self.scene.terrain.terrain_generator.num_rows = 20
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.12) # (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.5
        self.actions.joint_pos.joint_names = [
            "BL_collar_joint", "BL_hip_joint", "BL_knee_joint",
            "BR_collar_joint", "BR_hip_joint", "BR_knee_joint",
            "FL_collar_joint", "FL_hip_joint", "FL_knee_joint",
            "FR_collar_joint", "FR_hip_joint", "FR_knee_joint",
        ]
        self.actions.joint_pos.preserve_order = True

        # event
        self.events.physics_material.params["static_friction_range"] = (0.7, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.0)
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # commands
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.7, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # observations
        self.observations.policy.base_lin_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.3, n_max=0.3)
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.05, n_max=0.05)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-1.5, n_max=1.5)
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.1, n_max=0.1)
        # self.observations.policy.joint_pos.scale = 1.0
        # self.observations.policy.joint_vel.scale = 0.05
        # self.observations.policy.base_lin_vel.scale = 2.0
        # self.observations.policy.base_ang_vel.scale = 0.25

        # rewards
        self.rewards.track_lin_vel_xy_exp.weight = 2.0  # default 1.0, haraduka: 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.9  # default 0.5, haraduka: 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0  # default -2.0, haraduka: -10.0
        self.rewards.ang_vel_xy_l2.weight = -0.05  # default -0.05, haraduka: -0.1
        self.rewards.dof_torques_l2.weight = -1.0e-3  # default -1.0e-5, haraduka: -1.0e-3
        self.rewards.dof_acc_l2.weight = -1.0e-6  # default -2.5e-7, haraduka: -1.0e-4
        self.rewards.action_rate_l2.weight = -0.1  # default -0.01, haraduka: -0.1
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01  # default 0.125, haraduka: 0.001
        self.rewards.undesired_contacts = None
        # self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["base", ".*_thigh", ".*_scapula"]
        # self.rewards.undesired_contacts.weight = 0.0  # default -1.0, haraduka: None
        self.rewards.flat_orientation_l2.weight = -1.0  # default 0.0, haraduka: -10.0
        self.rewards.base_height_l2.weight = 0.00   # default -0.005, haraduka: None
        self.rewards.dof_pos_limits.weight = -10.0  # default 0.0, haraduka: -10.0
        self.rewards.dof_vel_l2.weight = -1.0e-7  # default None, haraduka: -1.0e-7

        self.rewards.stand_still.weight = -1.0  # default None, haraduka: -10.0
        self.rewards.feet_stumble.weight = -0.01  # default None, haraduka: None
        self.rewards.gait.weight = 0.2 # default None, haraduka: None

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base"]
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["base", ".*_thigh", ".*_scapula"]

import omni.isaac.lab.terrains as terrain_gen

@configclass
class MeviusRoughEnvCfg_PLAY(MeviusRoughEnvCfg):
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
        
        ####### additional change
        self.scene.terrain.terrain_generator.size = (8.0, 8.0)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.1, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.3
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.1, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.1, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.3
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["rails"] = terrain_gen.MeshRailsTerrainCfg(
            proportion=0.2, platform_width=4.0, size=(5.0, 5.0), rail_height_range=(0.1, 0.1), rail_thickness_range=(0.3, 0.4), 
        )

        #######

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
