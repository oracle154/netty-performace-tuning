#!/usr/bin/env bash
source venv/bin/activate

jmeter_results_root_folder="/home/wso2/supun/netty-performance"
parent_folder="netty/dataset/"

mkdir -p ${parent_folder}

duration="900"
warm_up="300"
users="100"
test="DbWrite" # Should be one of; DbWrite DbRead Prime10m Prime1m Prime10k Prime100k
metrics_window_size="60"
measuring_interval="10"

netty_host="192.168.32.11"
netty_port="15000"

for use_case in "DBRead" "DBWrite" "Prime1M"
do
    for users in 20 50 100 200 300
    do
        for threads in "tomcat" 20 50 100 200 300 500
        do
            case_name="${use_case}_${users}_${threads}"

            echo "Running ${case_name}"

            echo "Restarting Netty server"
            # restart netty
            ssh wso2@192.168.32.11 "./supun/scripts/restart-netty.sh ${test} ${threads} ${metrics_window_size}"

            echo "Reconnecting to Netty server (JMX connection)"
            # reconnect to netty server (JMX connection)
            curl http://192.168.32.2:8080/reconnect-netty

            echo "Start collecting server side metrics in background"
            nohup python3 netty_metrics.py ${parent_folder} ${case_name} ${warm_up} ${duration} 0 ${measuring_interval} ${metrics_window_size} > netty_metrics.log &

            echo "Starting JMeter client"
            # start apache JMeter
            ssh wso2@192.168.32.6 "./supun/jmeter/bin/jmeter -Jgroup1.host=${netty_host} -Jgroup1.port=${netty_port} -Jgroup1.threads=${users} -Jgroup1.duration=$((${duration}+${warm_up})) -n -t /home/wso2/supun/jmeter/bin/NettyTest.jmx -l ${jmeter_results_root_folder}/${parent_folder}/${case_name}.jtl"

            echo "Collecting stats summary from JTL Splitter"
            # collect client side summary from JTL
            ssh wso2@192.168.32.6 "java -jar /home/wso2/supun/jtl-spliter.jar -f ${jmeter_results_root_folder}/${parent_folder}/${case_name}.jtl -u SECONDS -t ${warm_up} -s"

            ssh wso2@192.168.32.6 "cat ${jmeter_results_root_folder}/${parent_folder}/${case_name}-measurement-summary.json" | python3 generate_client_summary.py ${parent_folder} ${case_name}

        done
    done
done
