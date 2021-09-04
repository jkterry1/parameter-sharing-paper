
import sys
import json
import numpy as np
import os
import pickle as pkl
import time
from pprint import pprint

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.utils import set_random_seed

from pettingzoo.butterfly import cooperative_pong_v3, prospector_v4, knights_archers_zombies_v7
from pettingzoo.atari import entombed_cooperative_v2, pong_v2, basketball_pong_v2

import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first

import optuna
from optuna.integration.skopt import SkoptSampler
from optuna.pruners import BasePruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

from utils.hyperparams_opt import sample_ppo_params, sample_dqn_params
from utils.callbacks import SaveVecNormalizeCallback, TrialEvalCallback

from indicator_util import AgentIndicatorWrapper, InvertColorIndicator, BinaryIndicator, GeometricPatternIndicator

import argparse

from stable_baselines3.common.utils import set_random_seed

if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser()
    
    '''
    Env List
    - Entombed Cooperative (Atari): DQN, PPO
    - Cooperative Pong (Butterfly): DQN, PPO
    - Prospector (Butterfly): PPO
    - KAZ (Butterfly): DQN, PPO
    - Pong (Atari): DQN, PPO
    - Basketball Pong (Atari): DQN, PPO
    '''
    butterfly_envs = ["prospector-v4", "knights-archers-zombies-v7", "cooperative-pong-v3"]
    atari_envs = ["entombed-cooperative-v2", "basketball-pong-v2", "pong-v2"]

    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=["ppo", "dqn"])
    parser.add_argument("--env", type=str, default="pong-v2", help="environment ID", choices=[
        "prospector-v4",
        "knights-archers-zombies-v7",
        "cooperative-pong-v3",
        "entombed-cooperative-v2",
        "basketball-pong-v2",
        "pong-v2"
    ])
    parser.add_argument("-n", "--n-timesteps", help="Overwrite the number of timesteps", default=1e6, type=int)
    parser.add_argument("--n-trials", help="Number of trials for optimizing hyperparameters", type=int, default=10)
    parser.add_argument(
        "--optimization-log-path",
        help="Path to save the evaluation log and optimal policy for each hyperparameter tried during optimization. "
        "Disabled if no argument is passed.",
        type=str,
    )
    parser.add_argument("--eval-episodes", help="Number of episodes to use for evaluation", default=5, type=int)
    parser.add_argument(
        "--sampler",
        help="Sampler to use when optimizing hyperparameters",
        type=str,
        default="tpe",
        choices=["random", "tpe", "skopt"],
    )
    parser.add_argument(
        "--pruner",
        help="Pruner to use when optimizing hyperparameters",
        type=str,
        default="median",
        choices=["halving", "median", "none"],
    )
    parser.add_argument("--n-startup-trials", help="Number of trials before using optuna sampler", type=int, default=10)
    parser.add_argument(
        "--n-evaluations",
        help="Training policies are evaluated every n-timesteps // n-evaluations steps when doing hyperparameter optimization",
        type=int,
        default=100,
    )
    parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default="logs")
    parser.add_argument(
        "--storage", help="Database storage path if distributed optimization should be used", type=str, default=None
    )
    parser.add_argument("--study-name", help="Study name for distributed optimization", type=str, default=None)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    args = parser.parse_args()

    seed = np.random.randint(2 ** 32 - 1, dtype="int64").item()
    set_random_seed(seed)

    print("=" * 10, args.env, "=" * 10)
    print(f"Seed: {seed}")
    
    # Hyperparameter optimization

    # Determine sampler and pruner
    if args.sampler == "random":
        sampler = RandomSampler(seed=seed)
    elif args.sampler == "tpe":
        sampler = TPESampler(n_startup_trials=args.n_startup_trials, seed=seed)
    elif args.sampler == "skopt":
        sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")
    
    if args.pruner == "halving":
        pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
    elif args.pruner == "median":
        pruner = MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=args.n_evaluations // 3)
    elif args.pruner == "none":
        # Do not prune
        pruner = MedianPruner(n_startup_trials=args.n_trials, n_warmup_steps=args.n_evaluations)
    else:
        raise ValueError(f"Unknown pruner: {args.pruner}")

    print(f"Sampler: {args.sampler} - Pruner: {args.pruner}")

    # Create study
    study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            storage=args.storage,
            study_name=args.study_name,
            load_if_exists=True,
            direction="maximize",
        )

    hyperparams_sampler = {'ppo': sample_ppo_params, 'dqn': sample_dqn_params}
    hyperparams_algo = {'ppo': PPO, 'dqn': DQN}
    
    muesli_obs_size = 96 
    muesli_frame_size = 4

    # Objective function for hyperparameter search
    def objective(trial: optuna.Trial) -> float:
        #kwargs = self._hyperparams.copy()
        kwargs = {
            #'n_envs': 1,
            'policy': 'CnnPolicy',
            #'n_timesteps': 1e6,
        }

        # Sample candidate hyperparameters
        sampled_hyperparams =  hyperparams_sampler[args.algo](trial)
        kwargs.update(sampled_hyperparams)

        # Create training env
        if args.env == "prospector-v4":
            env = prospector_v4.parallel_env()
            agent_type = "prospector"
        elif args.env == "knights-archers-zombies-v7":
            env = knights_archers_zombies_v7.parallel_env()
            agent_type = "archer"
        elif args.env == "cooperative-pong-v3":
            env = cooperative_pong_v3.parallel_env()
            agent_type = "paddle_0"
        elif args.env == "entombed-cooperative-v2":
            env = entombed_cooperative_v2.parallel_env()
            agent_type = "first"
        elif args.env == "basketball-pong-v2":
            env = basketball_pong_v2.parallel_env()
            agent_type = "first"
        elif args.env == "pong-v2":
            env = pong_v2.parallel_env()
            agent_type = "first"
        env = ss.color_reduction_v0(env)
        env = ss.resize_v0(env, x_size=muesli_obs_size, y_size=muesli_obs_size, linear_interp=True)
        env = ss.pad_action_space_v0(env)
        env = ss.pad_observations_v0(env)
        env = ss.frame_stack_v1(env, stack_size=muesli_frame_size)

        # Enable black death
        if args.env == 'knights-archers-zombies-v7':
            env = ss.black_death_v2(env)

        # Agent indicator wrapper
        agent_indicator_name = trial.suggest_categorical("agent_indicator", choices=["identity", "invert", "invert-replace", "binary", "geometric"])
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

        model =  hyperparams_algo[args.algo](
            env=env,
            tensorboard_log=None,
            # We do not seed the trial
            seed=None,
            verbose=0,
            **kwargs,
        )

        model.trial = trial

        # Create eval env
        if args.env == "prospector-v4":
            eval_env = prospector_v4.parallel_env()
            agent_type = "prospector"
        elif args.env == "knights-archers-zombies-v7":
            eval_env = knights_archers_zombies_v7.parallel_env()
            agent_type = "archer"
        elif args.env == "cooperative-pong-v3":
            eval_env = cooperative_pong_v3.parallel_env()
            agent_type = "paddle_0"
        elif args.env == "entombed-cooperative-v2":
            eval_env = entombed_cooperative_v2.parallel_env()
            agent_type = "first"
        elif args.env == "basketball-pong-v2":
            eval_env = basketball_pong_v2.parallel_env()
            agent_type = "first"
        elif args.env == "pong-v2":
            eval_env = pong_v2.parallel_env()
            agent_type = "first"
        eval_env = ss.color_reduction_v0(eval_env)
        eval_env = ss.resize_v0(eval_env, x_size=muesli_obs_size, y_size=muesli_obs_size, linear_interp=True)
        eval_env = ss.pad_action_space_v0(eval_env)
        eval_env = ss.pad_observations_v0(eval_env)
        eval_env = ss.frame_stack_v1(eval_env, stack_size=muesli_frame_size)
        # Enable black death
        if args.env == 'knights-archers-zombies-v7':
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

        optuna_eval_freq = int(args.n_timesteps / args.n_evaluations)
        # Account for parallel envs
        optuna_eval_freq = max(optuna_eval_freq // model.get_env().num_envs, 1)
        # Use non-deterministic eval for Atari
        path = None
        if args.optimization_log_path is not None:
            path = os.path.join(args.optimization_log_path, f"trial_{str(trial.number)}")
        #callbacks = get_callback_list({"callback": self.specified_callbacks})
        callbacks = []
        deterministic_eval = args.env not in atari_envs
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            best_model_save_path=path,
            log_path=path,
            n_eval_episodes=args.eval_episodes,
            eval_freq=optuna_eval_freq,
            deterministic=deterministic_eval,
        )
        callbacks.append(eval_callback)

        try:
            model.learn(args.n_timesteps, callback=callbacks)
            # Free memory
            model.env.close()
            eval_env.close()
        except (AssertionError, ValueError) as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            print("============")
            print("Sampled hyperparams:")
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned()
        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

    pass

    try:
        study.optimize(objective, n_trials=args.n_trials, n_jobs=1)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    report_name = (
        f"report_{args.env}_{args.n_trials}-trials-{args.n_timesteps}"
        f"-{args.sampler}-{args.pruner}_{int(time.time())}"
    )

    log_path = os.path.join(args.log_folder, args.algo, report_name)

    if args.verbose:
        print(f"Writing report to {log_path}")

    # Write report
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    study.trials_dataframe().to_csv(f"{log_path}.csv")

    # Save python object to inspect/re-use it later
    with open(f"{log_path}.pkl", "wb+") as f:
        pkl.dump(study, f)

    # Plot optimization result
    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)

        fig1.show()
        fig2.show()
    except (ValueError, ImportError, RuntimeError):
        pass