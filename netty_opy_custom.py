import sklearn.gaussian_process as gp
import numpy as np
import random
from scipy.stats import norm
from skopt.acquisition import gaussian_ei
import time
import requests
import sys
import csv
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin


def dummy_model(x):
    return 5*x[0]**2 - 4*x[1]*x[0] + 33 * x[1] + 334


def acquisition_function(x, model, minimum):
    x = np.array(x).reshape(1, -1)
    mu, sigma = model.predict(x, return_std=True)
    print(mu, sigma)
    with np.errstate(divide='ignore'):
        Z = (minimum - mu) / sigma
        print(norm.cdf(Z))
        expected_improvement = (minimum - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
        # expected_improvement[sigma == 0.0] = 0.0
    return -1 * expected_improvement


def _normalize(x, minimum, maximum):
    return (x - minimum) / (maximum - minimum)


def get_performance_only_tomcat(x, i):
    global data

    requests.put("http://192.168.32.2:8080/setThreadPoolNetty?size=" + str(x[0]))

    time.sleep((i+1) * tuning_interval + start_time - time.time())

    res = requests.get("http://192.168.32.2:8080/performance-netty").json()
    data.append(res)
    print("Mean response time : " + str(res[2]))
    return float(res[2])


def objective(x):
    global data
    global param_history
    global ii

    x = int(x)

    requests.put("http://192.168.32.2:8080/setThreadPoolNetty?size=" + str(x))
    param_history.append([x])

    time.sleep((ii+1) * tuning_interval + start_time - time.time())
    ii += 1
    res = requests.get("http://192.168.32.2:8080/performance-netty").json()
    data.append(res)
    print("Mean response time : " + str(res[2]))
    return float(res[2])


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
initial_points = 4


model = gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(), alpha=noise_level,
                                    n_restarts_optimizer=10, normalize_y=True)

x_data = []
y_data = []

start_time = time.time()
use_tpe = True

if use_tpe:
    ii = 0
    space = hp.uniform('x', 4, 200)
    tpe_trials = Trials()
    tpe_best = fmin(fn=objective, space=space, algo=tpe.suggest, trials=tpe_trials,
                    max_evals=test_duration // tuning_interval)

else:
    thread_pool_max = 200
    thread_pool_min = 4

    # sample more random (or predetermined data points) and collect numbers (up to initial points)
    for i in range(0, initial_points):
        x = thread_pool_min + i * (thread_pool_max-thread_pool_min) / initial_points
        x = int(x)
        x_data.append([_normalize(x, thread_pool_min, thread_pool_max)])
        y_data.append(get_performance_only_tomcat([x], i))
        param_history.append([x])

    model.fit(x_data, y_data)

    # use bayesian optimization
    for i in range(initial_points, iterations):
        minimum = min(y_data)
        # minimum = 99999
        max_expected_improvement = 0
        max_points = []
        max_points_unnormalized = []

        for pool_size in range(thread_pool_min, thread_pool_max + 1):
            x = [pool_size]
            x_normalized = [_normalize(x[0], thread_pool_min, thread_pool_max)]

            ei = gaussian_ei(np.array(x_normalized).reshape(1, -1), model, minimum)

            if ei > max_expected_improvement:
                max_expected_improvement = ei
                max_points = [x_normalized]
                max_points_unnormalized = [x]

            elif ei == max_expected_improvement:
                max_points.append(x_normalized)
                max_points_unnormalized.append(x)

        if max_expected_improvement == 0:
            print("WARN: Maximum expected improvement was 0. Most likely to pick a random point next")

        # select the point with maximum expected improvement
        # if there're multiple points with same ei, chose randomly
        idx = random.randint(0, len(max_points) - 1)
        next_x = max_points[idx]
        param_history.append(max_points_unnormalized[idx])
        next_y = get_performance_only_tomcat(max_points_unnormalized[idx], i)
        x_data.append(next_x)
        y_data.append(next_y)

        model = gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(), alpha=noise_level,
                                            n_restarts_optimizer=10, normalize_y=True)

        model.fit(x_data, y_data)

    print("minimum found : ", min(y_data))

with open(folder_name + case_name + "/results.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["IRR", "Request Count", "Mean Latency (for window)", "99th Latency"])
    for line in data:
        writer.writerow(line)

with open(folder_name + case_name + "/param_history.csv", "w") as f:
    writer = csv.writer(f)
    for line in param_history:
        writer.writerow(line)

