# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for mevius robots.

The following configuration parameters are available:

* :obj:`MEVIUS_CFG`: The mevius robot

Reference:

* https://github.com/haraduka/mevius

"""

# from omni.isaac.lab.sensors.camera.camera_cfg import CameraCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import DelayedPDActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

##
# Configuration - Actuators.
##

T_MOTOR_AK70_10_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*_collar_joint", ".*_hip_joint", ".*_knee_joint"],
    effort_limit=24.8,
    velocity_limit=20.9,  # [rad/s] = 200 [rpm]
    stiffness={".*_collar_joint": 30.0, ".*_hip_joint": 30.0, ".*_knee_joint": 20.0},
    damping={".*_collar_joint": 1.0, ".*_hip_joint": 1.0, ".*_knee_joint": 0.8},
    # stiffness={".*_collar_joint": 25.0, ".*_hip_joint": 25.0, ".*_knee_joint": 25.0},
    # damping={".*_collar_joint": 0.5, ".*_hip_joint": 0.5, ".*_knee_joint": 0.5},
    min_delay=int(0.01/0.005),  # 0.01-0.0025
    max_delay=int(0.03/0.005),  # 0.03+0.0075
)
"""Configuration for mevius Delayed PDActuator model."""

##
# Configuration - Articulation.
##
import os
MEVIUS_ASSETS_BASEPATH = os.path.dirname(__file__) 
MEVIUS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MEVIUS_ASSETS_BASEPATH}/data/usd/mevius.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33),  # x,y,z [m]
        joint_pos={  # = target angles [rad] when action = 0.0
            '[F,B]R_collar_joint': -0.1,
            '[F,B]L_collar_joint': 0.1,
            'F[R,L]_hip_joint': 0.8,
            'B[R,L]_hip_joint': 1.0,
            '.*knee_joint': -1.4,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={"legs": T_MOTOR_AK70_10_CFG},
)
"""Configuration of mevius robot using simple actuator config.

Note:
    Since we don't have a publicly available actuator network for ANYmal-D, we use the same network as ANYmal-C.
    This may impact the sim-to-real transfer performance.
"""


##
# Configuration - Sensors.
##
# MEVIUS_REALSENSE = CameraCfg(

# )
# """Configuration for the Realsense T265 mounted on the mevius robot's base."""
