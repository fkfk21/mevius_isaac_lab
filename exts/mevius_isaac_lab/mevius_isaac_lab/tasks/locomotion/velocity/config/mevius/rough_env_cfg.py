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
from mevius_isaac_lab.assets.mevius import MEVIUS_CFG, MEVIUS_JOINT_NAMES


@configclass
class MeviusRewardsCfg(RewardsCfg):
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=0.00,
        params={
            "command_name": "base_velocity",
            "cmd_lin_vel_threshold": 0.1,
            "cmd_ang_vel_threshold": 0.1,
            "body_lin_vel_threshold": 0.2,
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
    foot_rhythm = RewTerm(
        func=spot_mdp.air_time_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "mode_time": 0.4,
            "velocity_threshold": 0.1,
        }
    )
    foot_slip = RewTerm(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    air_time_variance = RewTerm(
        func=spot_mdp.air_time_variance_penalty,
        weight=-0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    
    def __post_init__(self):
        super().__post_init__()

        # set the parameters to match the hardware
        self.undesired_contacts.params["sensor_cfg"].body_names = [".*_thigh", ".*_calf"]
        self.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"

        # weights
        self.track_lin_vel_xy_exp.weight = 1.5
        self.track_ang_vel_z_exp.weight  = 0.9
        self.lin_vel_z_l2.weight         = -2.0
        self.ang_vel_xy_l2.weight        = -0.1
        self.dof_torques_l2.weight       = -1.0e-5
        self.dof_acc_l2.weight           = -2.0e-7
        self.action_rate_l2.weight       = -0.05
        self.feet_air_time.weight        = 0.05
        self.undesired_contacts.weight   = -1.0
        self.flat_orientation_l2.weight  = -1.0
        self.dof_pos_limits.weight       = -5.0
        self.dof_vel_l2.weight           = -1.0e-6
        self.stand_still.weight          = -2.0
        self.gait.weight                 = 0.3
        self.foot_rhythm.weight          = 0.2
        self.foot_slip.weight            = -0.3
        self.air_time_variance.weight    = -0.1


@configclass
class MeviusSceneCfg(MySceneCfg):

    def __post_init__(self):
        super().__post_init__()

        # set the robot as mevius
        self.robot = MEVIUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain parameter settings
        self.terrain.max_init_terrain_level = 5
        self.terrain.terrain_generator.num_rows = 20
        self.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.10)
        self.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.09)
        self.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.02, 0.20)
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.02, 0.20)

@configclass
class MeviusObservationsCfg(ObservationsCfg):

    @configclass
    class CommonCfg(ObservationsCfg.PolicyCfg):

        def __post_init__(self):
            super().__post_init__()

            # set the parameters to match the hardware
            self.joint_pos.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=MEVIUS_JOINT_NAMES, preserve_order=True)
            }
            self.joint_vel.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=MEVIUS_JOINT_NAMES, preserve_order=True)
            }

            # scale observations
            self.joint_pos.scale    = 1.0
            self.joint_vel.scale    = 0.05
            self.base_lin_vel.scale = 2.0
            self.base_ang_vel.scale = 0.25

            # clip observations
            self.base_lin_vel.clip      = (-100.0, 100.0)
            self.base_ang_vel.clip      = (-100.0, 100.0)
            self.velocity_commands.clip = (-100.0, 100.0)
            self.joint_pos.clip         = (-100.0, 100.0)
            self.joint_vel.clip         = (-100.0, 100.0)
            self.actions.clip           = ( -10.0,  10.0)

    @configclass
    class PolicyCfg(CommonCfg):
        height_scan = None

        def __post_init__(self):
            super().__post_init__()

            # add noise to the observations
            self.enable_corruption = True
            self.base_lin_vel.noise      = Unoise(n_min=-0.15, n_max=0.15)
            self.base_ang_vel.noise      = Unoise(n_min=-0.2, n_max=0.2)
            self.joint_pos.noise         = Unoise(n_min=-0.01, n_max=0.01)
            self.joint_vel.noise         = Unoise(n_min=-1.0, n_max=1.0)
            self.projected_gravity.noise = Unoise(n_min=-0.1, n_max=0.1)

    @configclass
    class CriticCfg(CommonCfg):
        ## add more privileged observations
        
        def __post_init__(self):
            super().__post_init__()

            # disable noise for critic observations
            self.enable_corruption = False

            # change offset for base height scan
            self.height_scan.params["offset"] = 0.3

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

    def __post_init__(self):
        return super().__post_init__()


@configclass
class MeviusRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: MeviusSceneCfg = MeviusSceneCfg(num_envs=2048, env_spacing=2.5)
    rewards: MeviusRewardsCfg = MeviusRewardsCfg()
    observations: MeviusObservationsCfg = MeviusObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # actions
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.joint_names = MEVIUS_JOINT_NAMES
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.clip = {".*": (-10.0, 10.0) }

        # events
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.0)
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
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
        self.events.push_robot.params["velocity_range"] = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
        }

        # commands
        # self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 2.0
        self.commands.base_velocity.ranges.lin_vel_x = (-0.9, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base",".*_scapula", ".*_thigh"]


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

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
