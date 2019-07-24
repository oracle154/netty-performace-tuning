#!/usr/bin/env bash

parent_folder="netty/tuning/"

mkdir -p ${parent_folder}

duration="900"
warm_up_1="150"
warm_up_2="150"
users="100"
test="DbWrite" # Should be one of (exact match) DbWrite DbRead Prime10m Prime1m Prime10k Prime100k
start_threads="100"
metrics_window_size="60"
measuring_interval="10"
tuning_interval="60"

netty_host="192.168.32.11"
netty_port="15000"

# restart netty
ssh wso2@192.168.32.11 "./supun/scripts/restart-netty.sh ${test} ${start_threads} ${metrics_window_size}"

# start apache jmeter
nohup ssh wso2@192.168.32.6 "./supun/jmeter/bin/jmeter -Jgroup1.host=192.168.32.11 -Jgroup1.port=15000 -Jgroup1.threads=${users} -Jgroup1.duration=$((${duration}+${warm_up_1}+${warm_up_2})) -n -t /home/wso2/supun/jmeter/bin/NettyTest.jmx -l fist_test.jtl" > jmeter.log &

# start collecting metrics
sleep ${warm_up_1}
curl http://192.168.32.2:9010/reconnect-netty
sleep ${warm_up_2}

nohup python3 netty_metrics.py ${parent_folder} "test_case_gp_tuning" 0 ${duration} 0 ${measuring_interval} ${metrics_window_size} > netty_metrics.log &
#python3 netty_metrics.py ${parent_folder} "test_case_fixed_100" 0 ${duration} 0 ${measuring_interval} ${metrics_window_size}

python3 netty_opy_custom.py ${parent_folder} "test_case_gp_tuning" 0 ${duration} 0 ${tuning_interval}
