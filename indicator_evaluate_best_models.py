# Evaluate stored best model and create a log of it
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from pettingzoo.butterfly import cooperative_pong_v3, prospector_v4, knights_archers_zombies_v7
from pettingzoo.atari import entombed_cooperative_v2, pong_v2
from pettingzoo.atari.base_atari_env import BaseAtariEnv, base_env_wrapper_fn, parallel_wrapper_fn
import supersuit as ss
import json
from indicator_util import AgentIndicatorWrapper, InvertColorIndicator, BinaryIndicator, GeometricPatternIndicator

from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage, VecNormalize
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", type=str, default=None)
parser.add_argument("--study-name", help="Study name used during hyperparameter optimization", type=str, default=None)
parser.add_argument("--parameter-id", type=int, default=None)
parser.add_argument("--n-runs", type=int, default=None)
args = parser.parse_args()

study_dir = './indicator_hyperparameters/' + args.study_name

param_id = args.parameter_id
eval_log_dir = study_dir + '/eval_logs/hyperparameter_' + str(param_id) + '/'

result_per_timestep = {}

EVAL_RUNS = 10

muesli_obs_size = 96 
muesli_frame_size = 4

# Load data
eval_log_file = eval_log_dir + 'reward_stat.txt'
param_id = args.parameter_id
with open(eval_log_file, "w+") as logf:
    all_mean_rewards = []
    for i in range(args.n_runs):
        # Construct eval env    
        param_file = study_dir + "/hyperparameters_" + str(param_id) + ".json"
        with open(param_file) as f:
            params = json.load(f)

        agent_indicator_name = params['agent_indicator']

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
        def image_transpose(env):
            if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
                env = VecTransposeImage(env)
            return env
        eval_env = image_transpose(eval_env)

        # ======================================================

        eval_run_log_dir = eval_log_dir + 'run_' + str(i) + '/'
        eval_run_best_model = eval_run_log_dir + 'best_model'

        model = PPO.load(eval_run_best_model)
        mean_reward, std_reward = evaluate_policy(model, eval_env, deterministic=True, n_eval_episodes=EVAL_RUNS * eval_env.num_envs)
        
        log = str(i) + "th mean reward:" + str(mean_reward) + " / std reward:" + str(std_reward)
        print(log)
        logf.write(log)
        logf.write('\n')
        all_mean_rewards.append(mean_reward)

    total_mean_reward = sum(all_mean_rewards) / len(all_mean_rewards)
    log = "Total mean reward:" + str(total_mean_reward)
    print(log)
    logf.write(log)