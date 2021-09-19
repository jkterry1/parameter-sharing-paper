mkdir -p ./eval_logs/kaz/
rm -rf eval_logs/kaz/*

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env knights-archers-zombies-v7 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name kaz_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/kaz_1_1e7_0 > ./eval_logs/kaz/eval_0.out &
sleep 3
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env knights-archers-zombies-v7 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name kaz_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/kaz_1_1e7_0 > ./eval_logs/kaz/eval_1.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env knights-archers-zombies-v7 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name kaz_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/kaz_1_1e7_0 > ./eval_logs/kaz/eval_2.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env knights-archers-zombies-v7 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name kaz_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/kaz_1_1e7_0 > ./eval_logs/kaz/eval_3.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env knights-archers-zombies-v7 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name kaz_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/kaz_1_1e7_0 > ./eval_logs/kaz/eval_4.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env knights-archers-zombies-v7 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name kaz_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/kaz_1_1e7_0 > ./eval_logs/kaz/eval_5.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env knights-archers-zombies-v7 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name kaz_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/kaz_1_1e7_0 > ./eval_logs/kaz/eval_6.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python indicator_opt.py --algo ppo --env knights-archers-zombies-v7 --n-timesteps 10000000 --n-trials 5 --n-evaluations 200 --sampler tpe --pruner median --study-name kaz_1_1e7_0 --storage mysql://shh1295:graphicboy1%21@13.125.14.92:3306/kaz_1_1e7_0 > ./eval_logs/kaz/eval_7.out &