# from gym.envs.registration import register

# register(
#     id='maTracking-v0',
#     entry_point='maTTenv.env:maTrackingEnv0',
# )
# register(
#     id='setTracking-v0',
#     entry_point='maTTenv.env:setTrackingEnv0',
# )


from utilities.ma_time_limit import maTimeLimit

def make(env_name, render=False, figID=0, record=False, ros=False, directory='',
                    T_steps=None, num_agents=2, num_targets=1, **kwargs):
    """
    env_name : str
        name of an environment. (e.g. 'Cartpole-v0')
    type : str
        type of an environment. One of ['atari', 'classic_control',
        'classic_mdp','target_tracking']
    """
    if T_steps is None:
        T_steps = 200

    if env_name == 'maTracking-v2':
        from maPEenv.env.maTracking_v2 import maTrackingEnv2
        env0 = maTrackingEnv2(num_agents=num_agents, num_targets=num_targets, **kwargs)

    elif env_name == 'maPDefense-v0':
        from maPEenv.env.maPDefense_v0 import maPDefenseEnv0
        env0 = maPDefenseEnv0(num_agents=num_agents, num_targets=num_targets, **kwargs)
    else:
        raise ValueError('No such environment exists.')

    env = maTimeLimit(env0, max_episode_steps=T_steps)

    if ros:
        from ttenv.ros_wrapper import Ros
        env = Ros(env)
    if render:
        from maPEenv.display_wrapper import Display2D
        env = Display2D(env, figID=figID)
    if record:
        from maPEenv.display_wrapper import Video2D
        env = Video2D(env, dirname = directory)

    return env