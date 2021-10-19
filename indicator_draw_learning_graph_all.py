# Draw learning graph of all hyperparameters
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

parser = argparse.ArgumentParser()
parser.add_argument("--study-name", help="Study name used during hyperparameter optimization", type=str, default=None)
parser.add_argument("--parameter-id", type=int, default=None)
parser.add_argument("--n-runs", type=int, default=None)
args = parser.parse_args()

study_dir = './indicator_hyperparameters/' + args.study_name

result_per_timestep_per_hyperparam = {}

for h in range(args.parameter_id):
    param_id = h
    if param_id == 0:
        continue

    eval_log_dir = study_dir + '/eval_logs/hyperparameter_' + str(param_id) + '/'

    result_per_timestep = {}

    # Load data
    for i in range(args.n_runs):    
        eval_run_log_dir = eval_log_dir + 'run_' + str(i) + '/'
        eval_run_log = eval_run_log_dir + 'evaluations.npz'

        data = np.load(eval_run_log)
        data_timesteps = data['timesteps']
        data_results = data['results']

        if len(result_per_timestep.keys()) == 0:
            for t in range(len(data_timesteps)):
                data_timestep = data_timesteps[t]
                data_result = data_results[t]

                # Store mean of 10 evaluations for each run.
                result_per_timestep[data_timestep] = np.mean(data_result)
        else:
            for t in range(len(data_timesteps)):
                data_timestep = data_timesteps[t]
                data_result = data_results[t]

                if data_timestep not in result_per_timestep.keys():
                    print("Inconsistent time step error")
                    exit()

                result_per_timestep[data_timestep] = np.append(result_per_timestep[data_timestep], np.mean(data_result))

    result_per_timestep_per_hyperparam[param_id] = result_per_timestep

# Draw graph
fig, ax = plt.subplots()
fig.set_size_inches(16, 12)
clrs = sns.color_palette("husl", args.parameter_id)
with sns.axes_style("darkgrid"):
    timesteps = list(result_per_timestep_per_hyperparam[1].keys())
    
    for i in range(args.parameter_id):
        if i not in result_per_timestep_per_hyperparam.keys():
            continue
        result_per_timestep = result_per_timestep_per_hyperparam[i]
        nrow = len(timesteps)
        ncol = args.n_runs
        results = np.zeros((nrow, ncol))
        for j in range(nrow):
            results[j][:] = result_per_timestep[timesteps[j]]
    
        mean_results = np.mean(results, axis = 1)
        std_results = np.std(results, axis = 1)

        mean_spline = make_interp_spline(timesteps, mean_results)
        std_spline = make_interp_spline(timesteps, std_results)

        n_timesteps = np.linspace(0, np.max(timesteps), 500)
        n_mean_results = mean_spline(n_timesteps)
        n_std_results = std_spline(n_timesteps)

        ax.plot(n_timesteps, n_mean_results, label="Hyperparameter " + str(i), c = clrs[i])
        ax.fill_between(n_timesteps, n_mean_results - n_std_results, n_mean_results + n_std_results, alpha=0.3, facecolor=clrs[i])
        ax.legend()

    #plt.show()
    plt.savefig(study_dir + '/aggregate_learning_graph.png')