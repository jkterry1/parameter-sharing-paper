mkdir -p ./eval_logs/prospector/
rm -rf eval_logs/prospector/*

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env prospector-v4 --n-timesteps 100000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name prospector_1_1e8_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/prospector_1_1e8_0 > ./eval_logs/prospector/eval_0.out &
sleep 3
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env prospector-v4 --n-timesteps 100000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name prospector_1_1e8_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/prospector_1_1e8_0 > ./eval_logs/prospector/eval_1.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env prospector-v4 --n-timesteps 100000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name prospector_1_1e8_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/prospector_1_1e8_0 > ./eval_logs/prospector/eval_2.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env prospector-v4 --n-timesteps 100000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name prospector_1_1e8_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/prospector_1_1e8_0 > ./eval_logs/prospector/eval_3.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env prospector-v4 --n-timesteps 100000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name prospector_1_1e8_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/prospector_1_1e8_0 > ./eval_logs/prospector/eval_4.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env prospector-v4 --n-timesteps 100000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name prospector_1_1e8_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/prospector_1_1e8_0 > ./eval_logs/prospector/eval_5.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env prospector-v4 --n-timesteps 100000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name prospector_1_1e8_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/prospector_1_1e8_0 > ./eval_logs/prospector/eval_6.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env prospector-v4 --n-timesteps 100000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name prospector_1_1e8_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/prospector_1_1e8_0 > ./eval_logs/prospector/eval_7.out &