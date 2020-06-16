import numpy as np

# Based on ttenv METADATA_v2
METADATA_v0={   # Extremely random inits for targets and beliefs
        'version' : 0,
        'sensor_r': 10.0,
        'fov' : 150,
        'sensor_r_sd': 0.2, # sensor range noise.
        'sensor_b_sd': 0.01, # sensor bearing noise.
        'target_init_cov': 30.0, # initial target diagonal Covariance.
        'target_init_vel': 0.0, # target's initial velocity.
        'target_vel_limit': 2.0, # velocity limit of targets.
        'init_distance_min': 5.0, # the minimum distance btw targets and the agent.
        'init_distance_max': 15.0, # the maximum distance btw targets and the agent.
        'init_belief_distance_min': 0.5, # the minimum distance btw targets and the agent.
        'init_belief_distance_max': 15.0, # the maximum distance btw targets and the agent.
        'margin': 1.0, # a marginal distance btw targets and the agent.
        'margin2wall': 0.5, # a marginal distance from a wall.
        'action_v': [3, 2, 1, 0], # action primitives - linear velocities.
        'action_w': [np.pi/2, 0, -np.pi/2], # action primitives - angular velocities.
        'const_q': 0.001, # target noise constant in beliefs.
        'const_q_true': 0.01, # target noise constant of actual targets.
    }

METADATA_v1={   # Beliefs are initialized near target
        'version' : 1,
        'sensor_r': 10.0,
        'fov' : 90,
        'sensor_r_sd': 0.2, # sensor range noise.
        'sensor_b_sd': 0.01, # sensor bearing noise.
        'target_init_cov': 30.0, # initial target diagonal Covariance.
        'target_init_vel': 0.0, # target's initial velocity.
        'target_vel_limit': 2.0, # velocity limit of targets.
        'init_distance_min': 5.0, # the minimum distance btw targets and the agent.
        'init_distance_max': 15.0, # the maximum distance btw targets and the agent.
        'init_belief_distance_min': 1.0, # the minimum distance btw belief and the target.
        'init_belief_distance_max': 7.5, # the maximum distance btw belief and the target.
        'margin': 1.0, # a marginal distance btw targets and the agent.
        'margin2wall': 0.5, # a marginal distance from a wall.
        'action_v': [2, 1.33, 0.67, 0], # action primitives - linear velocities.
        'action_w': [np.pi/2, 0, -np.pi/2], # action primitives - angular velocities.
        'const_q': 0.001, # target noise constant in beliefs.
        'const_q_true': 0.01, # target noise constant of actual targets.
    }

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

# Designate a metadata version to be used throughout the target tracking env.
METADATA = METADATA_v2