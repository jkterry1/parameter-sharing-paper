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

# Load data
for trial in range(args.n_runs):
    eval_run_log_dir = eval_log_dir + 'run_' + str(trial) + '/'
    eval_run_log = eval_run_log_dir + 'evaluations.npz'

    data = np.load(eval_run_log)
    data_timesteps = data['timesteps']
    data_results = data['results']

    data_mean_results = np.mean(data_results, axis=1)
    data_std_results = np.std(data_results, axis=1)

    # Draw graph
    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", 1)
    with sns.axes_style("darkgrid"):
        mean_spline = make_interp_spline(data_timesteps, data_mean_results)
        std_spline = make_interp_spline(data_timesteps, data_std_results)

        n_timesteps = np.linspace(0, np.max(data_timesteps), 500)
        n_mean_results = mean_spline(n_timesteps)
        n_std_results = std_spline(n_timesteps)

        ax.plot(n_timesteps, n_mean_results, label="Hyperparameter " + str(args.parameter_id) + " Run " + str(trial), c = clrs[0])
        ax.fill_between(n_timesteps, n_mean_results - n_std_results, n_mean_results + n_std_results, alpha=0.3, facecolor=clrs[0])
        ax.legend()

        #plt.show()
        plt.savefig(eval_run_log_dir + 'learning_graph.png')