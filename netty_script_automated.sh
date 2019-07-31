#!/usr/bin/env bash

jmeter_results_root_folder="/home/wso2/supun/netty-performance"
parent_folder="netty/tuning/"

mkdir -p ${parent_folder}

duration="900"
warm_up_1="300"
warm_up_2="300"
warm_up="600"
users="300"
test="Prime1m" # Should be one of (exact match) DbWrite DbRead Prime10m Prime1m Prime10k Prime100k
start_threads="100"
metrics_window_size="60"
measuring_interval="10"
tuning_interval="60"
baysien_def="true_Thread_200"

netty_host="192.168.32.11"
netty_port="15000"

for users in 20 50 100 200 300 500
do

    case_name="${test}_${users}_${baysien_def}"

    echo "Running ${case_name}"

    echo "Restarting Netty server"

    # restart netty
    ssh wso2@192.168.32.11 "./supun/scripts/restart-netty.sh ${test} ${start_threads} ${metrics_window_size}"

    echo "Starting JMeter client"
    # start apache jmeter
    ssh wso2@192.168.32.6 "./supun/jmeter/bin/jmeter -Jgroup1.host=192.168.32.11 -Jgroup1.port=15000 -Jgroup1.threads=${users} -Jgroup1.duration=$((${duration}+${warm_up_1}+${warm_up_2})) -n -t /home/wso2/supun/jmeter/bin/NettyTest.jmx -l fist_test.jtl" > jmeter.log &

    # start collecting metrics
    sleep ${warm_up_1}
    curl http://192.168.32.2:8080/reconnect-netty
    sleep ${warm_up_2}

    echo "Start collecting server side metrics in background"
    python3 netty_metrics.py ${parent_folder} ${case_name} 0 ${duration} 0 ${measuring_interval} ${metrics_window_size} > netty_metrics.log &

    python3 netty_opy_custom.py ${parent_folder} ${case_name} 0 ${duration} 0 ${tuning_interval}
    # start apache jmeter
    jobs
    wait

    #ssh wso2@192.168.32.6 "java -jar /home/wso2/supun/jtl-spliter.jar -f ${jmeter_results_root_folder}/${parent_folder}/${case_name}.jtl -u SECONDS -t ${warm_up} -s" > jslipter.log &

done