#!/bin/bash

# 定义日志文件路径
LOG_FILE="dr3+vlm_script.log"

while true; do
    # 执行 Python 程序，并将标准输出和标准错误输出重定向到日志文件
    python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk333 >> $LOG_FILE 2>&1
    # 将提示信息追加到日志文件
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 程序退出，即将重新启动..." >> $LOG_FILE
    sleep 1  # 可以根据需要调整等待时间
done