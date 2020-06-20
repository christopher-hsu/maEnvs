import numpy as np


METADATA_v2={   #Beliefs are initialized near target, easier init than v1
        'version' : 1,
        'sensor_r': 10.0,
        'fov' : 90,
        'sensor_r_sd': 0.2, # sensor range noise.
        'sensor_b_sd': 0.01, # sensor bearing noise.
        'target_init_cov': 30.0, # initial target diagonal Covariance.
        'target_init_vel': 0.0, # target's initial velocity.
        'target_vel_limit': 2.0, # velocity limit of targets.
        'init_distance_min': 5.0, # the minimum distance btw targets and the agent.
        'init_distance_max': 10.0, # the maximum distance btw targets and the agent.
        'init_belief_distance_min': 0.0, # the minimum distance btw belief and the target.
        'init_belief_distance_max': 5.0, # the maximum distance btw belief and the target.
        'margin': 1.0, # a marginal distance btw targets and the agent.
        'margin2wall': 0.5, # a marginal distance from a wall.
        'action_v': [2, 1.33, 0.67, 0], # action primitives - linear velocities.
        'action_w': [np.pi/2, 0, -np.pi/2], # action primitives - angular velocities.
        'const_q': 0.001, # target noise constant in beliefs.
        'const_q_true': 0.01, # target noise constant of actual targets.
    }

METADATA_multi_v1={
        'version' : 'm1',
        'sensor_r': 10.0,
        'fov' : 120,
        'sensor_r_sd': 0.2, # sensor range noise.
        'sensor_b_sd': 0.01, # sensor bearing noise.
        'target_init_cov': 30.0, # initial target diagonal Covariance.
        'target_init_vel': [0.0, 0.0], # target's initial velocity.
        'target_speed_limit': 1.0, # velocity limit of targets.
        'lin_dist_range_a2b':(5.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 10.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'margin': 1.0, # a marginal distance btw targets and the agent.
        'margin2wall': 1.0, # a marginal distance from a wall.
        'action_v': [3, 2, 1, 0], # action primitives - linear velocities.
        'action_w': [np.pi/2, 0, -np.pi/2], # action primitives - angular velocities.
        'const_q': 0.01, # target noise constant in beliefs.
        'const_q_true': 0.01, # target noise constant of actual targets.
    }

METADATA_pd={   #Beliefs are initialized near target
        'version' : 1,
        'sensor_r': 10.0,
        'fov' : 90,
        'sensor_r_sd': 0.2, # sensor range noise.
        'sensor_b_sd': 0.01, # sensor bearing noise.
        'agent_init_dist_min': 15.0, # the minimum distance btw agent and the origin.
        'agent_init_dist_max': 15.0, # the maximum distance btw agent and the origin.
        'target_init_cov': 30.0, # initial target diagonal Covariance.
        'target_init_vel': [0.0, 0.0], # target's initial velocity.
        'target_vel_limit': 2.0, # velocity limit of targets.
        'target_init_dist_min': 40.0, # the minimum distance btw targets and the origin.
        'target_init_dist_max': 40.0, # the maximum distance btw targets and the origin.
        'init_belief_distance_min': 0.0, # the minimum distance btw belief and the target.
        'init_belief_distance_max': 5.0, # the maximum distance btw belief and the target.
        'margin': 1.0, # a marginal distance btw targets and the agent.
        'margin2wall': 0.5, # a marginal distance from a wall.
        'action_v': [2, 1.33, 0.67, 0], # action primitives - linear velocities.
        'action_w': [np.pi/2, 0, -np.pi/2], # action primitives - angular velocities.
        'const_q': 0.001, # target noise constant in beliefs.
        'const_q_true': 0.01, # target noise constant of actual targets.
        'spiral_min': 0.001, # minimum factor of target spiral
        'spiral_max': 0.01  # maximum factor of target spiral
    }

# Designate a metadata version to be used throughout the target tracking env.
METADATA = METADATA_pd