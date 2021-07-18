CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python train.py --algo ppo --env FetchReach-v1 -optimize --n-trials 100 --sampler tpe --pruner median --study-name FetchReach_v1 --storage mysql://shh1295:graphicboy1%21@13.125.226.90/FetchReach_v1 > FetchReach_v1_console_0.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python train.py --algo ppo --env FetchReach-v1 -optimize --n-trials 100 --sampler tpe --pruner median --study-name FetchReach_v1 --storage mysql://shh1295:graphicboy1%21@13.125.226.90/FetchReach_v1 > FetchReach_v1_console_1.out &
sleep 3