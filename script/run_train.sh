#!/bin/bash

# 后台运行命令，并将输出重定向到指定文件
nohup mpirun -n 4 python CPM-mindspore/train.py > CPM-mindspore/log_four_cards.log 2>&1 &

# 打印后台进程的 PID（可选）
echo "后台进程的 PID: $!"