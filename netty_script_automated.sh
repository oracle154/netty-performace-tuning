#!/usr/bin/env bash

jmeter_results_root_folder="/home/wso2/supun/netty-performance"
parent_folder="netty/tuning/"

mkdir -p ${parent_folder}

duration="1800"
warm_up_1="150"
warm_up_2="150"
warm_up="300"
users="300"
test="DBWrite" # Should be one of (exact match) DbWrite DbRead Prime10m Prime1m Prime10k Prime100k
start_threads="100"
metrics_window_size="30"
measuring_interval="10"
tuning_interval="30"
baysien_def="false_Thread_200_99th"

netty_host="192.168.32.11"
netty_port="15000"

online_check="True"

#for users in 20 50 100 200 300 500
for users in 10 20 50 100 200 300 500
do

    case_name="${test}_${users}_${baysien_def}"

    echo "Running ${case_name}"

    echo "Restarting Netty server"

    # restart netty
    ssh wso2@192.168.32.11 "./supun/scripts/restart-netty.sh ${test} ${start_threads} ${metrics_window_size}"
    echo "Starting JMeter client"
    # start apache jmeter
    #This line if for steady concurrent useres
    ssh wso2@192.168.32.6 "./supun/jmeter/bin/jmeter -Jgroup1.host=192.168.32.11 -Jgroup1.port=15000 -Jgroup1.threads=${users} -Jgroup1.duration=$((${duration}+${warm_up_1}+${warm_up_2})) -n -t /home/wso2/supun/jmeter/bin/NettyTest_original.jmx -l ${jmeter_results_root_folder}/${parent_folder}/${case_name}.jtl" > jmeter.log &

    # start collecting metrics
    echo "connecting to netty"
    sleep ${warm_up_1}
    echo "sleep 1 is done"
    curl http://192.168.32.2:8080/reconnect-netty
    sleep ${warm_up_2}

    echo "Start collecting server side metrics in background"
    python3 netty_metrics_2.py ${parent_folder} ${case_name} 0 ${duration} 0 ${measuring_interval} ${metrics_window_size} > netty_metrics.log &

    #python3 bayesian_both.py ${parent_folder} ${case_name} 0 ${duration} 0 ${tuning_interval}
    python3 bayesian_both.py ${parent_folder} ${case_name} 0 ${duration} 0 ${tuning_interval} ${online_check}
    #python3 netty_opy_custom.py ${parent_folder} ${case_name} 0 ${duration} 0 ${tuning_interval}
    # start apache jmeter
    jobs
    wait

    echo "Collecting stats summary from JTL Splitter"
    # collect client side summary from JTL
    ssh wso2@192.168.32.6 "java -jar /home/wso2/supun/jtl-spliter.jar -f ${jmeter_results_root_folder}/${parent_folder}/${case_name}.jtl -u SECONDS -t ${warm_up} -s"

    ssh wso2@192.168.32.6 "cat ${jmeter_results_root_folder}/${parent_folder}/${case_name}-measurement-summary.json" | python3 generate_client_summary.py ${parent_folder} ${case_name}


    #ssh wso2@192.168.32.6 "java -jar /home/wso2/supun/jtl-spliter.jar -f ${jmeter_results_root_folder}/${parent_folder}/${case_name}.jtl -u SECONDS -t ${warm_up} -s" > jslipter.log &

done