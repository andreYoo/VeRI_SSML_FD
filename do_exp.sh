#!/bin/bash

#Training script
#python tools/train.py --experiments experiments/veri776.yml --gpus 0,1 --suffix 'veri776'
#python tools/train.py --experiments experiments/veri-wild.yml --gpus 0,1 --suffix 'veri-wild'
#python tools/train_with_cpu.py --experiments ./experiments/veri-wild.yml --suffix 'test' --use_dram True #Train veri-wild with CPU and DRAM

#TEST script
python tools/test.py --experiments experiments/veri776.yml --gpus 0,1
python tools/test.py --experiments experiments/veri-wild.yml --gpus 0,1
