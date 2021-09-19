mkdir -p ./eval_logs/entombed_cooperative/
rm -rf eval_logs/entombed_cooperative/*

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env entombed-cooperative-v2 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name entombed_cooperative_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/entombed_cooperative_1_1e7_0 > ./eval_logs/entombed_cooperative/eval_0.out &
sleep 3
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env entombed-cooperative-v2 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name entombed_cooperative_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/entombed_cooperative_1_1e7_0 > ./eval_logs/entombed_cooperative/eval_1.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env entombed-cooperative-v2 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name entombed_cooperative_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/entombed_cooperative_1_1e7_0 > ./eval_logs/entombed_cooperative/eval_2.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env entombed-cooperative-v2 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name entombed_cooperative_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/entombed_cooperative_1_1e7_0 > ./eval_logs/entombed_cooperative/eval_3.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env entombed-cooperative-v2 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name entombed_cooperative_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/entombed_cooperative_1_1e7_0 > ./eval_logs/entombed_cooperative/eval_4.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env entombed-cooperative-v2 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name entombed_cooperative_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/entombed_cooperative_1_1e7_0 > ./eval_logs/entombed_cooperative/eval_5.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env entombed-cooperative-v2 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name entombed_cooperative_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/entombed_cooperative_1_1e7_0 > ./eval_logs/entombed_cooperative/eval_6.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env entombed-cooperative-v2 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name entombed_cooperative_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/entombed_cooperative_1_1e7_0 > ./eval_logs/entombed_cooperative/eval_7.out &