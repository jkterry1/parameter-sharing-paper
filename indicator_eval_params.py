import sys
import json
import argparse
import os

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

from pettingzoo.butterfly import cooperative_pong_v3, prospector_v4, knights_archers_zombies_v7
from pettingzoo.atari import entombed_cooperative_v2, pong_v2
from pettingzoo.atari.base_atari_env import BaseAtariEnv, base_env_wrapper_fn, parallel_wrapper_fn

import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first

from indicator_util import AgentIndicatorWrapper, InvertColorIndicator, BinaryIndicator, GeometricPatternIndicator


from torch import nn as nn
from typing import Dict, Any
from datetime import datetime

n_evaluations = 100
n_envs = 1

parser = argparse.ArgumentParser()
parser.add_argument("--study-name", help="Study name used during hyperparameter optimization", type=str, default=None)
parser.add_argument("--env-name", help="Env to use during hyperaparameter evaluation", type=str, default=None)
parser.add_argument("--parameter-id", type=int, default=None)
parser.add_argument("--n-runs", type=int, default=None)
parser.add_argument("--timesteps", type=int, default=1e7)
args = parser.parse_args()
n_timesteps = args.timesteps

study_dir = './indicator_hyperparameters/' + args.study_name
start_time = datetime.now()

muesli_obs_size = 96 
muesli_frame_size = 4

for param_id in range(1):
    param_id = args.parameter_id
    param_file = study_dir + "/hyperparameters_" + str(param_id) + ".json"
    with open(param_file) as f:
        params = json.load(f)

    if(params['batch_size'] > params['n_steps']):
        params['batch_size'] = params['n_steps']

    print("Evaluating Hyperparameters...")
    print(params)

    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[params['net_arch']]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[params['activation_fn']]

    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=activation_fn,
        ortho_init=False,
    )
    agent_indicator_name = params['agent_indicator']

    del(params['net_arch'])
    del(params['activation_fn'])
    del(params['agent_indicator'])
    params['policy_kwargs'] = policy_kwargs
    params['policy'] = 'CnnPolicy'

    # Generate env
    if args.env_name == "prospector-v4":
        env = prospector_v4.parallel_env()
        agent_type = "prospector"
    elif args.env_name == "knights-archers-zombies-v7":
        env = knights_archers_zombies_v7.parallel_env()
        agent_type = "archer"
    elif args.env_name == "cooperative-pong-v3":
        env = cooperative_pong_v3.parallel_env()
        agent_type = "paddle_0"
    elif args.env_name == "entombed-cooperative-v2":
        env = entombed_cooperative_v2.parallel_env()
        agent_type = "first"
    elif args.env_name == "pong-v2":
        env = pong_v2.parallel_env()
        agent_type = "first"
    env = ss.color_reduction_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pad_observations_v0(env)
    env = ss.resize_v0(env, x_size=muesli_obs_size, y_size=muesli_obs_size, linear_interp=True)
    env = ss.frame_stack_v1(env, stack_size=muesli_frame_size)

    # Enable black death
    if args.env_name == 'knights-archers-zombies-v7':
        env = ss.black_death_v2(env)

    # Agent indicator wrapper
    if agent_indicator_name == "invert":
        agent_indicator = InvertColorIndicator(env, agent_type)
        agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator)
    elif agent_indicator_name == "invert-replace":
        agent_indicator = InvertColorIndicator(env, agent_type)
        agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator, False)
    elif agent_indicator_name == "binary":
        agent_indicator = BinaryIndicator(env, agent_type)
        agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator)
    elif agent_indicator_name == "geometric":
        agent_indicator = GeometricPatternIndicator(env, agent_type)
        agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator)
    if agent_indicator_name != "identity":
        env = ss.observation_lambda_v0(env, agent_indicator_wrapper.apply, agent_indicator_wrapper.apply_space)

    env = ss.pettingzoo_env_to_vec_env_v0(env)
    #env = ss.concat_vec_envs_v0(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
    env = VecMonitor(env)

    def image_transpose(env):
        if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
            env = VecTransposeImage(env)
        return env
    env = image_transpose(env)

    # Generate eval env
    if args.env_name == "prospector-v4":
        eval_env = prospector_v4.parallel_env()
        agent_type = "prospector"
    elif args.env_name == "knights-archers-zombies-v7":
        eval_env = knights_archers_zombies_v7.parallel_env()
        agent_type = "archer"
    elif args.env_name == "cooperative-pong-v3":
        eval_env = cooperative_pong_v3.parallel_env()
        agent_type = "paddle_0"
    elif args.env_name == "entombed-cooperative-v2":
        eval_env = entombed_cooperative_v2.parallel_env()
        agent_type = "first"
    elif args.env_name == "pong-v2":
        def pong_single_raw_env(**kwargs):
            return BaseAtariEnv(game="pong", num_players=1, env_name=os.path.basename(__file__)[:-3], **kwargs)
        pong_single_env = base_env_wrapper_fn(pong_single_raw_env)
        pong_parallel_env = parallel_wrapper_fn(pong_single_env)
        eval_env = pong_parallel_env()
        #eval_env = pong_v2.parallel_env()
        #eval_env = gym.make("Pong-v0", obs_type='image')
        agent_type = "first"
    eval_env = ss.color_reduction_v0(eval_env)
    eval_env = ss.pad_action_space_v0(eval_env)
    eval_env = ss.pad_observations_v0(eval_env)
    eval_env = ss.resize_v0(eval_env, x_size=muesli_obs_size, y_size=muesli_obs_size, linear_interp=True)
    eval_env = ss.frame_stack_v1(eval_env, stack_size=muesli_frame_size)
    # Enable black death
    if args.env_name == 'knights-archers-zombies-v7':
        eval_env = ss.black_death_v2(eval_env)

    # Agent indicator wrapper
    if agent_indicator_name == "invert":
        eval_agent_indicator = InvertColorIndicator(eval_env, agent_type)
        eval_agent_indicator_wrapper = AgentIndicatorWrapper(eval_agent_indicator)
    elif agent_indicator_name == "invert-replace":
        eval_agent_indicator = InvertColorIndicator(eval_env, agent_type)
        eval_agent_indicator_wrapper = AgentIndicatorWrapper(eval_agent_indicator, False)
    elif agent_indicator_name == "binary":
        eval_agent_indicator = BinaryIndicator(eval_env, agent_type)
        eval_agent_indicator_wrapper = AgentIndicatorWrapper(eval_agent_indicator)
    elif agent_indicator_name == "geometric":
        eval_agent_indicator = GeometricPatternIndicator(eval_env, agent_type)
        eval_agent_indicator_wrapper = AgentIndicatorWrapper(eval_agent_indicator)
    if agent_indicator_name != "identity":
        eval_env = ss.observation_lambda_v0(eval_env, eval_agent_indicator_wrapper.apply, eval_agent_indicator_wrapper.apply_space)

    eval_env = ss.pettingzoo_env_to_vec_env_v0(eval_env)
    #eval_env = ss.concat_vec_envs_v0(eval_env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
    eval_env = VecMonitor(eval_env)
    eval_env = image_transpose(eval_env)

    eval_freq = int(n_timesteps / n_evaluations)

    all_mean_rewards = []
    eval_log_dir = study_dir + '/eval_logs/hyperparameter_' + str(param_id) + '/'
    os.makedirs(eval_log_dir, exist_ok=True)

    eval_log_file = eval_log_dir + 'reward_stat.txt'
    with open(eval_log_file, "w+") as f:
        for i in range(args.n_runs):
            model = PPO(
                env=env,
                tensorboard_log=None,
                # We do not seed the trial
                seed=None,
                verbose=2,
                **params,
            )

            eval_run_log_dir = eval_log_dir + 'run_' + str(i)
            
            eval_callback = EvalCallback(eval_env, n_eval_episodes=10, best_model_save_path=eval_run_log_dir, log_path=eval_run_log_dir, eval_freq=eval_freq, deterministic=True, render=False)
            model.learn(total_timesteps=n_timesteps, callback=eval_callback)
            model = PPO.load(eval_run_log_dir + '/best_model')
            mean_reward, std_reward = evaluate_policy(model, eval_env, deterministic=True, n_eval_episodes=10)
            
            log = str(i) + "th mean reward:" + str(mean_reward) + " / std reward:" + str(std_reward)
            print(log)
            f.write(log)
            f.write('\n')
            all_mean_rewards.append(mean_reward)
        
        total_mean_reward = sum(all_mean_rewards) / len(all_mean_rewards)
        log = "Total mean reward:" + str(total_mean_reward)
        print(log)
        f.write(log)

end_time = datetime.now()
print("Elapsed Time: ")
print(end_time - start_time)
