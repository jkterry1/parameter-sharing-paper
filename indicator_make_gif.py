from stable_baselines3 import PPO
from pettingzoo.butterfly import cooperative_pong_v3, prospector_v4, knights_archers_zombies_v7
from pettingzoo.atari import entombed_cooperative_v2, pong_v2
import supersuit as ss

from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
import numpy as np
from indicator_util import AgentIndicatorWrapper, InvertColorIndicator, BinaryIndicator, GeometricPatternIndicator

from PIL import Image
import matplotlib.pyplot as plt

import argparse
import json

n_evaluations = 1

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", type=str, default=None)
parser.add_argument("--study-name", help="Study name used during hyperparameter optimization", type=str, default=None)
parser.add_argument("--parameter-id", type=int, default=None)
parser.add_argument("--run-id", type=int, default=None)
args = parser.parse_args()

def image_transpose(env):
    if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
        env = VecTransposeImage(env)
    return env

muesli_obs_size = 96 
muesli_frame_size = 4

# Load the best model
avg_rewards = {}
for param_id in range(args.parameter_id):
    for run_id in range(args.run_id):
        study_dir = './indicator_hyperparameters/' + args.study_name
        best_model_path = study_dir + '/eval_logs/hyperparameter_' + str(param_id) + '/run_' + str(run_id) + '/best_model.zip'
        best_model = PPO.load(best_model_path)

        # Construct render env        
        param_file = study_dir + "/hyperparameters_" + str(param_id) + ".json"
        with open(param_file) as f:
            params = json.load(f)
        agent_indicator_name = params['agent_indicator']

        if args.env_name == "prospector-v4":
            render_env = prospector_v4.env()
            agent_type = "prospector"
        elif args.env_name == "knights-archers-zombies-v7":
            render_env = knights_archers_zombies_v7.env()
            agent_type = "archer"
        elif args.env_name == "cooperative-pong-v3":
            render_env = cooperative_pong_v3.env()
            agent_type = "paddle_0"
        elif args.env_name == "entombed-cooperative-v2":
            render_env = entombed_cooperative_v2.env()
            agent_type = "first"
        elif args.env_name == "pong-v2":
            render_env = pong_v2.env()
            agent_type = "first"
        render_env = ss.color_reduction_v0(render_env)
        render_env = ss.pad_action_space_v0(render_env)
        render_env = ss.pad_observations_v0(render_env)
        render_env = ss.resize_v0(render_env, x_size=muesli_obs_size, y_size=muesli_obs_size, linear_interp=True)
        render_env = ss.frame_stack_v1(render_env, stack_size=muesli_frame_size)
        if args.env_name == 'knights-archers-zombies-v7':
            render_env = ss.black_death_v2(render_env)

        if agent_indicator_name == "invert":
            agent_indicator = InvertColorIndicator(render_env, agent_type)
            agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator)
        elif agent_indicator_name == "invert-replace":
            agent_indicator = InvertColorIndicator(render_env, agent_type)
            agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator, False)
        elif agent_indicator_name == "binary":
            agent_indicator = BinaryIndicator(render_env, agent_type)
            agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator)
        elif agent_indicator_name == "geometric":
            agent_indicator = GeometricPatternIndicator(render_env, agent_type)
            agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator)
        if agent_indicator_name != "identity":
            render_env = ss.observation_lambda_v0(render_env, agent_indicator_wrapper.apply, agent_indicator_wrapper.apply_space)

        for run in range(n_evaluations):
            obs_list = []
            i = 0
            render_env.reset()

            current_rewards = np.zeros(len(render_env.possible_agents))

            while True:
                for agent in render_env.agent_iter():
                    observation, reward, done, _ = render_env.last()
                    current_rewards[i % len(render_env.possible_agents)] += reward
                    action = best_model.predict(observation)[0] if not done else None

                    render_env.step(action)
                    i += 1
                    if i % (len(render_env.possible_agents)) == 0:
                        obs_list.append(render_env.render(mode='rgb_array'))
                avg_reward = sum(current_rewards) / len(current_rewards)
                avg_rewards["{}_{}".format(param_id, run_id)] = avg_reward
                render_env.close()
                break
            print('Writing gif {}'.format(run))
            imgs = [Image.fromarray(img) for img in obs_list]
            gif_path = study_dir + '/eval_logs/best_model_gifs_with_rewards/' + '/hyperparameter_' + str(param_id) + "_run_" + str(run_id) + "_{}_".format(avg_reward) + '.gif'
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=60, loop=0)

sorted_avg_rewards = {k: v for k, v in sorted(avg_rewards.items(), key=lambda item: item[1], reverse=True)}
eval_log_file = study_dir + '/eval_logs/best_model_gifs_with_rewards/avg_rewards.txt'
with open(eval_log_file, "w+") as logf:
    for key in sorted_avg_rewards.keys():
        logf.write(key + ": " + str(sorted_avg_rewards[key]) + "\n")
print(sorted_avg_rewards)