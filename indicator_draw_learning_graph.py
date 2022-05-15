# Draw learning graph of single hyperparameter
import argparse

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

param_id = args.parameter_id
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

# Draw graph
fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 1)
with sns.axes_style("darkgrid"):
    timesteps = list(result_per_timestep.keys())
    
    nrow = len(timesteps)
    ncol = args.n_runs
    results = np.zeros((nrow, ncol))
    for i in range(nrow):
        results[i][:] = result_per_timestep[timesteps[i]]
    
    mean_results = np.mean(results, axis = 1)
    std_results = np.std(results, axis = 1)

    mean_spline = make_interp_spline(timesteps, mean_results)
    std_spline = make_interp_spline(timesteps, std_results)

    n_timesteps = np.linspace(0, np.max(timesteps), 500)
    n_mean_results = mean_spline(n_timesteps)
    n_std_results = std_spline(n_timesteps)

    ax.plot(n_timesteps, n_mean_results, label="Hyperparameter " + str(args.parameter_id), c = clrs[0])
    ax.fill_between(n_timesteps, n_mean_results - n_std_results, n_mean_results + n_std_results, alpha=0.3, facecolor=clrs[0])
    ax.legend()

    #plt.show()
    plt.savefig(eval_log_dir + 'learning_graph.png')