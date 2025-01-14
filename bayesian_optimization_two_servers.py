import sklearn.gaussian_process as gp
import numpy as np
import random
# from skopt.acquisition import gaussian_ei
from acqstion import gaussian_ei, gaussian_pi, gaussian_lcb
import matplotlib.pyplot as plt
import time
import requests
import sys
import csv
import logging

logging.basicConfig(level=logging.INFO)

####################
np.random.seed(42)

kernel = gp.kernels.Matern()

def get_performance(x_pass, loc):
    global data

    requests.put("http://192.168.32.2:8080/setThreadPoolNetty?size=" + str(x_pass[0], x_pass[1]))

    time.sleep((loc + 1) * tuning_interval + start_time - time.time())

    res = requests.get("http://192.168.32.2:8080/performance-netty").json()

    data.append(res)
    logging.info("99th Percentile :" + str(res[3]))
    return float(res[3])

def get_initial_points():
    iter_num = int(np.sqrt(number_of_initial_points))
    x_temp = []
    for i in range(0, iter_num + 1):
        x1 = thread_pool_min + i * (thread_pool_max - thread_pool_min) / iter_num
        for j in range(0, iter_num + 1):
            x2 = thread_pool_min + j * (thread_pool_max - thread_pool_min) / iter_num
            x_temp.append(int(x1))
            x_temp.append(int(x2))
            x_data.append(x_temp)
            y_data.append(get_performance(x_temp, i))
            param_history.append(x_temp)
            x_temp = []

def gausian_model(kern, xx, yy):
    model = gp.GaussianProcessRegressor(kernel=kern, alpha=noise_level,
                                        n_restarts_optimizer=10, normalize_y=True)

    model.fit(xx, yy)

    return model



folder_name = sys.argv[1] if sys.argv[1][-1] == "/" else sys.argv[1] + "/"
case_name = sys.argv[2]

ru = int(sys.argv[3])
mi = int(sys.argv[4])
rd = int(sys.argv[5])
tuning_interval = int(sys.argv[6])

data = []
param_history = []
test_duration = ru + mi + rd
iterations = test_duration // tuning_interval

noise_level = 1e-6
number_of_initial_points = 4

x_data = []
y_data = []

start_time = time.time()

thread_pool_max = 200
thread_pool_min = 4


get_initial_points()

model = gausian_model(kernel, x_data, y_data)

xi = 0.1

# use bayesian optimization
iteration_start_num = int(pow((np.sqrt(number_of_initial_points) + 1), 2))
for i in range(iteration_start_num, iterations):
    minimum = min(y_data)
    minimum = np.array(minimum)
    x_location = y_data.index(min(y_data))
    max_expected_improvement = 0
    max_points = []

    logging.info("xi - %f", xi)
    logging.info("iter - %i", i)

    for pool_size_1 in range(thread_pool_min, thread_pool_max + 1):
        for pool_size_2 in range(thread_pool_min, thread_pool_max + 1):
            x = [pool_size_1, pool_size_2]
            x_val = [x]

            # may be add a condition to stop explorering the already expored locations
            ei = gaussian_ei(np.array(x_val).reshape(1, -1), model, minimum, xi)

            if ei > max_expected_improvement:
                max_expected_improvement = ei
                max_points = [x_val]

            elif ei == max_expected_improvement:
                max_points.append(x_val)

    if max_expected_improvement == 0:
        logging.info("WARN: Maximum expected improvement was 0. Most likely to pick a random point next")
        next_x = x_data[x_location]
        logging.info(next_x)
        xi = xi - xi / 10
        if xi < 0.00001:
            xi = 0
    else:
        # select the point with maximum expected improvement
        # if there're multiple points with same ei, chose randomly
        idx = random.randint(0, len(max_points) - 1)
        next_x = max_points[idx]
        next_x = next_x[0]
        xi = xi + xi / 8
        if xi > 0.01:
            xi = 0.01
        elif xi == 0:
            xi = 0.00002

    logging.info(next_x)

    param_history.append(next_x)
    next_y = get_performance(next_x, i)
    y_data.append(next_y)
    x_data.append(next_x)

    x_data_arr = np.array(x_data)
    y_data_arr = np.array(y_data)

    model = gausian_model(kernel, x_data, y_data_arr)

logging.info("minimum found : %f", min(y_data))

with open(folder_name + case_name + "/results.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["IRR", "Request Count", "Mean Latency (for window)", "99th Latency"])
    for line in data:
        writer.writerow(line)

with open(folder_name + case_name + "/param_history.csv", "w") as f:
    writer = csv.writer(f)
    for line in param_history:
        writer.writerow(line)