import os, copy, pdb
import numpy as np
from numpy import linalg as LA
from gym import spaces, logger
from maPDenv.maps import map_utils
import maPDenv.util as util 
from maPDenv.agent_models import *
from maPDenv.belief_tracker import * #KFbelief
from maPDenv.metadata import METADATA
from maPDenv.policies import *
from maPDenv.env.maPDefense_Base import maPDefenseBase

"""
Perimeter Defense Environments for Reinforcement Learning.
[Variables]

d: radial coordinate of a belief target in the learner frame
alpha : angular coordinate of a belief target in the learner frame
ddot : radial velocity of a belief target in the learner frame
alphadot : angular velocity of a belief target in the learner frame
Sigma : Covariance of a belief target
o_d : linear distance to the closet obstacle point
o_alpha : angular distance to the closet obstacle point

[Environment Description]
Varying number of agents, varying number of randomly moving targets
No obstacles

maPDefenseEnv0 : SE2 Target model with UKF belief tracker
    obs state: [d, alpha, logdet(Sigma), observed, o_d, o_alpha] *nb_targets
            where nb_targets and nb_agents vary between a range
            num_targets describes the upperbound on possible number of targets in env
            num_agents describes the upperbound on possible number of agents in env
    Target : SE2 model [x,y,theta] + a control policy u=[v,w]
    Belief Target : UKF for SE2 model [x,y,theta]

"""

class maPDefenseEnv0(maPDefenseBase):

    def __init__(self, num_agents=1, num_targets=2, map_name='empty', 
                        is_training=True, known_noise=True, **kwargs):
        super().__init__(num_agents=num_agents, num_targets=num_targets,
                        map_name=map_name, is_training=is_training)

        self.id = 'maPDefense-v0'
        self.nb_agents = num_agents #only for init, will change with reset()
        self.nb_targets = num_targets #only for init, will change with reset()
        self.agent_dim = 3
        self.target_dim = 3
        self.target_init_vel = np.array(METADATA['target_init_vel'])
        # LIMIT
        self.limit = {} # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        rel_vel_limit = METADATA['target_vel_limit'] + METADATA['action_v'][0] # Maximum relative speed
        self.limit['state'] = np.array([[0.0, -np.pi, -50.0, 0.0, 0.0, -np.pi ],
                                        [600.0, np.pi, 50.0, 2.0, self.sensor_r, np.pi]])
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)
        self.target_noise_cov = METADATA['const_q'] * self.sampling_period * np.eye(self.target_dim)

        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = METADATA['const_q_true'] * \
                                self.sampling_period * np.eye(self.target_dim)

        # Build a robot 
        self.setup_agents()
        # Build a target
        self.setup_targets()
        self.setup_belief_targets()
        # Use custom reward
        self.get_reward()

    def setup_agents(self):
        self.agents = [AgentSE2(agent_id = 'agent-' + str(i), 
                        dim=self.agent_dim, sampling_period=self.sampling_period, 
                        limit=self.limit['agent'], 
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                        for i in range(self.num_agents)]

    def setup_targets(self):
        self.targets = [AgentSE2(agent_id = 'target-' + str(i),
                        dim=self.target_dim, sampling_period=self.sampling_period, 
                        limit=self.limit['target'],
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x),
                        policy=SpiralPolicy(self.sampling_period, self.MAP.origin, 
                                            METADATA['spiral_min'], METADATA['spiral_max']))
                        for i in range(self.num_targets)]

    def setup_belief_targets(self):
        self.belief_targets = [UKFbelief(agent_id = 'target-' + str(i),
                        dim=self.target_dim, limit=self.limit['target'], 
                        fx=SE2Dynamics,
                        W=self.target_noise_cov, 
                        obs_noise_func=self.observation_noise,
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                        for i in range(self.num_targets)]


    def get_reward(self, obstacles_pt=None, observed=None, is_training=True):
        return reward_fun(self.nb_targets, self.belief_targets, is_training)

    def reset(self,**kwargs):
        """
        Random initialization a number of agents and targets at the reset of the env epsiode.
        Agents are given random positions in the map, targets are given random positions near a random agent.
        Return an observation state dict with agent ids (keys) that refer to their observation
        """
        try: 
            self.nb_agents = kwargs['nb_agents']
            self.nb_targets = kwargs['nb_targets']
        except:
            self.nb_agents = np.random.random_integers(1, self.num_agents)
            self.nb_targets = np.random.random_integers(1, self.num_targets)
        obs_dict = {}
        init_pose = self.get_init_pose(**kwargs)
        # Initialize agents
        for ii in range(self.nb_agents):
            self.agents[ii].reset(init_pose['agents'][ii])
            obs_dict[self.agents[ii].agent_id] = []

        # Initialize targets and beliefs
        for nn in range(self.nb_targets):
            self.belief_targets[nn].reset(
                        init_state=init_pose['belief_targets'][nn],
                        init_cov=self.target_init_cov)
            self.targets[nn].reset(init_pose['targets'][nn])
        # For nb agents calculate belief of targets assigned
        for jj in range(self.nb_targets):
            for kk in range(self.nb_agents):
                r, alpha = util.relative_distance_polar(self.belief_targets[jj].state[:2],
                                            xy_base=self.agents[kk].state[:2], 
                                            theta_base=self.agents[kk].state[2])
                logdetcov = np.log(LA.det(self.belief_targets[jj].cov))
                obs_dict[self.agents[kk].agent_id].append([r, alpha, logdetcov, 0.0, 
                                                            self.sensor_r, np.pi])
        for agent_id in obs_dict:
            obs_dict[agent_id] = np.asarray(obs_dict[agent_id])
        return obs_dict

    def step(self, action_dict):
        obs_dict = {}
        reward_dict = {}
        done_dict = {'__all__':False}
        info_dict = {}

        # Targets move (t -> t+1)
        for n in range(self.nb_targets):
            self.targets[n].update() 
        # Agents move (t -> t+1) and observe the targets
        for ii, agent_id in enumerate(action_dict):
            obs_dict[self.agents[ii].agent_id] = []
            reward_dict[self.agents[ii].agent_id] = []
            done_dict[self.agents[ii].agent_id] = []

            action_vw = self.action_map[action_dict[agent_id]]

            # Locations of targets and agents in order to maintain a margin between them
            margin_pos = [t.state[:2] for t in self.targets[:self.nb_targets]]
            for p, ids in enumerate(action_dict):
                if agent_id != ids:
                    margin_pos.append(np.array(self.agents[p].state[:2]))
            _ = self.agents[ii].update(action_vw, margin_pos)
            
            observed = []
            # Update beliefs of targets
            for jj in range(self.nb_targets):
                # Observe
                obs = self.observation(self.targets[jj], self.agents[ii])
                observed.append(obs[0])
                self.belief_targets[jj].update(obs[0], obs[1], self.agents[ii].state,
                                            np.array([np.random.random(),
                                            np.pi*np.random.random()-0.5*np.pi]))

            # obstacles_pt = map_utils.get_closest_obstacle(self.MAP, self.agents[ii].state)

            # if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)
            # Calculate beliefs on only assigned targets
            for kk in range(self.nb_targets):
                r_b, alpha_b = util.relative_distance_polar(self.belief_targets[kk].state[:2],
                                        xy_base=self.agents[ii].state[:2], 
                                        theta_base=self.agents[ii].state[-1])
                obs_dict[agent_id].append([r_b, alpha_b,
                                        np.log(LA.det(self.belief_targets[kk].cov)), 
                                        float(observed[kk]), obstacles_pt[0], obstacles_pt[1]])
            obs_dict[agent_id] = np.asarray(obs_dict[agent_id])
        # Get all rewards after all agents and targets move (t -> t+1)
        reward, done, mean_nlogdetcov = self.get_reward(obstacles_pt, observed, self.is_training)
        reward_dict['__all__'], done_dict['__all__'], info_dict['mean_nlogdetcov'] = reward, done, mean_nlogdetcov
        return obs_dict, reward_dict, done_dict, info_dict

def reward_fun(nb_targets, belief_targets, is_training=True, c_mean=0.1):
    detcov = [LA.det(b_target.cov) for b_target in belief_targets[:nb_targets]]
    r_detcov_mean = - np.mean(np.log(detcov))# - np.std(np.log(detcov))
    reward = c_mean * r_detcov_mean

    mean_nlogdetcov = None
    if not(is_training):
        logdetcov = [np.log(LA.det(b_target.cov)) for b_target in belief_targets[:nb_targets]]
        mean_nlogdetcov = -np.mean(logdetcov)
    return reward, False, mean_nlogdetcov