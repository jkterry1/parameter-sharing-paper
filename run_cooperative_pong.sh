mkdir -p ./eval_logs/cooperative_pong/
rm -rf eval_logs/cooperative_pong/*

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env cooperative-pong-v3 --n-timesteps 4000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name cooperative_pong_1_4e6_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/cooperative_pong_1_4e6_0 > ./eval_logs/cooperative_pong/eval_0.out &
sleep 3
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env cooperative-pong-v3 --n-timesteps 4000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name cooperative_pong_1_4e6_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/cooperative_pong_1_4e6_0 > ./eval_logs/cooperative_pong/eval_1.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env cooperative-pong-v3 --n-timesteps 4000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name cooperative_pong_1_4e6_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/cooperative_pong_1_4e6_0 > ./eval_logs/cooperative_pong/eval_2.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env cooperative-pong-v3 --n-timesteps 4000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name cooperative_pong_1_4e6_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/cooperative_pong_1_4e6_0 > ./eval_logs/cooperative_pong/eval_3.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env cooperative-pong-v3 --n-timesteps 4000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name cooperative_pong_1_4e6_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/cooperative_pong_1_4e6_0 > ./eval_logs/cooperative_pong/eval_4.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env cooperative-pong-v3 --n-timesteps 4000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name cooperative_pong_1_4e6_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/cooperative_pong_1_4e6_0 > ./eval_logs/cooperative_pong/eval_5.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env cooperative-pong-v3 --n-timesteps 4000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name cooperative_pong_1_4e6_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/cooperative_pong_1_4e6_0 > ./eval_logs/cooperative_pong/eval_6.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env cooperative-pong-v3 --n-timesteps 4000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name cooperative_pong_1_4e6_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/cooperative_pong_1_4e6_0 > ./eval_logs/cooperative_pong/eval_7.out &