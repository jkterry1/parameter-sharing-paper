import optuna
import json
import numpy as np
import argparse
import os

from optuna.visualization import plot_optimization_history, plot_param_importances

parser = argparse.ArgumentParser()
parser.add_argument("--study-name", help="Study name used during hyperparameter optimization", type=str, default=None)
parser.add_argument("--storage", help="Database storage path used during hyperparameter optimization", type=str, default=None)
parser.add_argument("--save-n-best-hyperparameters", help="Save the hyperparameters for the n best trials that resulted in the best returns", type=int, default=0)
parser.add_argument("--visualize", help="Visualize the study results", type=bool, default=True)
args = parser.parse_args()

output_dir = "./indicator_hyperparameters/" + args.study_name
os.makedirs(output_dir, exist_ok=True)

study = optuna.create_study(study_name=args.study_name, storage=args.storage, load_if_exists=True, direction="maximize")
if args.visualize:
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.write_image(output_dir + "/optimization_history.png")
    fig2.write_image(output_dir + "/param_importances.png")

values = []
for trial in study.trials:
    values.append(trial.value)

scratch_values = [-np.inf if i is None else i for i in values]
ordered_indices = np.argsort(scratch_values)[::-1]

for i in range(args.save_n_best_hyperparameters):
    params = study.trials[ordered_indices[i]].params
    text = json.dumps(params)
    jsonFile = open(output_dir + '/hyperparameters_' + str(i) + ".json", "w+")
    jsonFile.write(text)
    jsonFile.close()