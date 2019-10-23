import sklearn.gaussian_process as gp
import numpy as np
import random
# from skopt.acquisition import gaussian_ei
# from acqstion import gaussian_ei, gaussian_pi, gaussian_lcb
from acqstion import gaussian_ei
import matplotlib.pyplot as plt
import time

# from bayesian_optimization_util import plot_approximation, plot_acquisition
from bayesian_optimization_util import plot_approximation

np.random.seed(42)

####################

# Boundary for the simulation data
bounds = np.array([[1, 181]])

# Define the Kernel for gaussian process
kernel = gp.kernels.Matern()


# variables
data = []
X_plot_data = []
Y_plot_data = []
param_history = []

x_data = []
y_data = []

# test Duration selection
test_duration = 1000
tuning_interval = 10
iterations = test_duration // tuning_interval

# level of noise for gaussian
noise_level = 1e-6

# number of initial points for the gaussian
number_of_initial_points = 8

# start a timer
start_time = time.time()

# bounds for the gaussian
thread_pool_max = 180
thread_pool_min = 4

#############################################
size = thread_pool_max + 1 - thread_pool_min
#count = np.zeros(size)
###########################################

# gaussian noise distribution for the Data point
noise_dist = np.random.normal(0, 5, 20)

# Value needs to be changed for the exploration or exploitation
xi = 0.1

# sampling basic FIFO variables

maximum_in_sampler = 5
remove_num = maximum_in_sampler - 1

dif = int(thread_pool_max - thread_pool_min + 1)

history = np.zeros((dif, 2))

for i in range(0, dif):
    history[i][0] = int(i + thread_pool_min)

'''
# Function to predict
def function(X):
    return -1.0*np.sin(X/10.0)*X
'''


# Concept drift functions
def function(x_value, fun_num):
    if fun_num == 0:
        return -1.0 * np.sin(x_value / 10.0) * x_value
    elif fun_num == 1:
        return -1.0 * np.cos(x_value / 5.0) * x_value
    elif fun_num == 2:
        return -2.0 * np.sin(x_value / 10.0) * x_value
    else:
        return -2.0 * np.sin(x_value / 20.0) * x_value


# plotting of initial function
def initial_plot():
    global X_plot_data
    X_plot_data = np.arange(bounds[:, 0], bounds[:, 1], 1).reshape(-1, 1)
    global Y_plot_data
    Y_plot_data = function(X_plot_data, 0)

    plt.plot(X_plot_data, Y_plot_data, lw=2, label='Noise-free objective')
    plt.legend()
    plt.show()


# get the values when the x values are set by bayesian
def get_performance(x_pass):
    global data
    noise_loc = np.random.randint(0, 19)
    x_data_loc = np.where(X_plot_data == x_pass)
    return_val = Y_plot_data[x_data_loc] + noise_dist[noise_loc]
    # return_val = Y_plot_data[x_data_loc]
    return return_val


# get the initial points
def get_initial_points():
    for i_point in range(0, (number_of_initial_points + 1)):
        x = thread_pool_min + i_point * (thread_pool_max - thread_pool_min) / number_of_initial_points
        x = int(x)
        x_data.append([x])
        y_data.append(get_performance([x]))
        param_history.append([x])

        # sampling count starts
        if x in history:
            loc_history = (np.where(history == x)[0])[0]
            history[loc_history][1] += 1
        else:
            print("ERROR : ", x, " is not in the range")


# plot the gaussian model with new data points
def data_plot():
    plot_approximation(model, X_plot_data, Y_plot_data, param_history, y_data, next_x, show_legend=i == 0)
    plt.title(f'Iteration {i + 1}')
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()

# gaussian model calculation with new data
def gaussian_model(kern, xx, yy):
    model_gaussian = gp.GaussianProcessRegressor(kernel=kern, alpha=noise_level,
                                        n_restarts_optimizer=10, normalize_y=True)
    model_gaussian.fit(xx, yy)
    return model_gaussian


# call initial functions
initial_plot()
get_initial_points()

# fit initial data to gaussian model
model = gaussian_model(kernel, x_data, y_data)


# ---------Don't know why ------------
plot_number = 0
keep_min = 10

if plot_number == keep_min:
    print(keep_min)

# to get the concept drift change
function_change = iterations / 4

# ---------------------------------

# use bayesian optimization
for i in range((number_of_initial_points + 1), iterations):

    if i == function_change * (plot_number + 1):
        plot_number += 1

    Y_plot_data = function(X_plot_data, plot_number)

    minimum = min(y_data)
    x_location = y_data.index(min(y_data))
    max_expected_improvement = 0
    max_points = []

    print("xi -", xi)
    print("iteration -", i)

    for pool_size in range(thread_pool_min, thread_pool_max + 1):
        x_val = [pool_size]
        # may be add a condition to stop explorering the already expored locations
        feed_val = np.array(x_val).reshape(1, -1)
        #ei = gaussian_ei(np.array(x_val).reshape(1, -1), model, minimum, xi)
        ei = gaussian_ei(feed_val, model, minimum, xi)

        if ei > max_expected_improvement:
            max_expected_improvement = ei
            max_points = [x_val]

        elif ei == max_expected_improvement:
            max_points.append(x_val)

        #else:
            #print("WARN: Expected improvement < Max value")

    if max_expected_improvement == 0:
        print("WARN: Maximum expected improvement was 0. Most likely to pick a random point next")
        # Don't know why this code is here------------------------------
        if keep_min < 10:
            idx = random.randint(0, len(max_points) - 1)
            next_x = max_points[idx]
            keep_min += 1
            xi = 0.1
            print("keep min - ", keep_min)
        else:
            next_x = x_data[x_location]
            xi = xi - xi / 10
            if xi < 0.00001:
                xi = 0
        # ---------------------------------------------------------
    else:
        # select the point with maximum expected improvement
        # if there're multiple points with same ei, chose randomly
        idx = random.randint(0, len(max_points) - 1)
        next_x = max_points[idx]
        xi = xi + xi / 8
        if xi > 0.01:
            xi = 0.01
        elif xi == 0:
            xi = 0.00002

    print("Next - X", next_x)

    #next_x = [6]

    # sampling FIFO
    if next_x in history:

        '''# selecting the location in the history
        location_in_history = np.where(history == next_x)
        for k in range(len(location_in_history[0])):
            print("K val", k)
            if location_in_history[1][k] == 0:
                loc_history = location_in_history[0][k]
                break
        '''
        loc_history = (np.where(history == next_x)[0])[0]

        if history[loc_history][1] == 0:
            history[loc_history][1] += 1
            for j in range(len(param_history)):
                if next_x <= param_history[j]:
                    param_history.insert(j, next_x)
                    next_y = get_performance(next_x)
                    y_data.insert(j, next_y)
                    x_data.insert(j, next_x)
                    break
        else:
            loc_num = param_history.index(next_x)

            param_history.insert(loc_num, next_x)
            next_y = get_performance(next_x)
            y_data.insert(loc_num, next_y)
            x_data.insert(loc_num, next_x)

            if history[loc_history][1] < maximum_in_sampler:
                history[loc_history][1] += 1
            else:
                param_history.remove(param_history[loc_num + maximum_in_sampler])
                x_data.remove(x_data[loc_num + maximum_in_sampler])
                y_data.remove(y_data[loc_num + maximum_in_sampler])

            max_num = int(history[loc_history][1])

            variance_matrix = []
            var_x = []

            for var_i in range(loc_num, loc_num + max_num):
                variance_matrix.append(y_data[var_i])
                var_x.append(x_data[var_i])


            variance = np.var(variance_matrix)
            print("variance", variance)

            if variance > 100:
                var_remove = loc_num + max_num
                while var_remove > (loc_num+1):
                    param_history.remove(param_history[var_remove])
                    x_data.remove(x_data[var_remove])
                    y_data.remove(y_data[var_remove])
                    history[loc_history][1] = 1
                    var_remove -= 1

                xi = 0.1

                '''for var_remove in range(loc_num+1, loc_num+max_num):
                    param_history.remove(param_history[var_remove])
                    x_data.remove(x_data[var_remove])
                    y_data.remove(y_data[var_remove])
                    history[loc_history][1] = 1
                xi = 0.1'''


        '''elif history[loc_history][1] < maximum_in_sampler:
            history[loc_history][1] += 1
            for j in range(len(param_history)):
                if next_x <= param_history[j]:
                    param_history.insert(j, next_x)
                    next_y = get_performance(next_x)
                    y_data.insert(j, next_y)
                    x_data.insert(j, next_x)
                    break
        else:
            #history[loc_history][1] = maximum_in_sampler
            if next_x in param_history:
                loc_num = param_history.index(next_x)

                variance_matrix = []

                for var_i in range (loc_num, loc_num+maximum_in_sampler):
                    variance_matrix.append(y_data[var_i])

                variance = np.var(variance_matrix)
                print("variance", variance)
        '''
        '''     y_difference_5 = abs(y_data[loc_num + remove_num] - next_y)
                y_difference_1 = abs(y_data[loc_num] - next_y)
                if y_difference_5 > abs(y_data[loc_num + remove_num] * 0.5):
                    keep_min =  0
                    if y_difference_1 < abs(y_data[loc_num] * 0.5):
                        param_history.remove(param_history[loc_num + remove_num - 1])
                        x_data.remove(x_data[loc_num + remove_num - 1])
                        y_data.remove(y_data[loc_num + remove_num - 1])

                        param_history.remove(param_history[loc_num + remove_num - 2])
                        x_data.remove(x_data[loc_num + remove_num - 2])
                        y_data.remove(y_data[loc_num + remove_num - 2])

                        param_history.remove(param_history[loc_num + remove_num - 3])
                        x_data.remove(x_data[loc_num + remove_num - 3])
                        y_data.remove(y_data[loc_num + remove_num - 3])

                        history[loc_history][1] = maximum_in_sampler - 3
                #else:'''
        '''        param_history.remove(param_history[loc_num + remove_num])
                x_data.remove(x_data[loc_num + remove_num])
                y_data.remove(y_data[loc_num + remove_num])

                param_history.insert(loc_num, next_x)
                next_y = get_performance(next_x)
                y_data.insert(loc_num, next_y)
                x_data.insert(loc_num, next_x)'''
    else:
        print("EEROR - values is not in the range")

    #########################################################c

    '''
    #####################################################################

    for j in range(len(param_history)):
        if next_x < param_history[j]:
            param_history.insert(j,next_x)
            next_y = get_performance(next_x, i)
            y_data.insert(j,next_y)
            x_data.insert(j,next_x)
            print(x_data)
            break
    '''
    #######################################################################
    '''
    #####################################################################

    if next_x in param_history: #and len(param_history) > 20:
        same_loc = param_history.index(next_x)
        count[same_loc] = count[same_loc] + 1
        next_y = get_performance(next_x, i)

        num = int(count[same_loc] + 1)
        number_of_data_points = 10

        k = np.random.randint(0, num)

        print("K = ", k)
        print(num)
        if k < number_of_data_points:
            y_data[same_loc] = next_y

    else:
        param_history.append(next_x)
        next_y = get_performance(next_x, i)
        y_data.append(next_y)
        x_data.append(next_x)

    #######################################################################'''

    # param_history.append(next_x)
    # next_y = get_performance(next_x, i)
    # y_data.append(next_y)
    # x_data.append(next_x)

    x_data_arr = np.array(x_data)
    y_data_arr = np.array(y_data)

    print("X data size = ", np.size(x_data))
    print("Y data size = ", np.size(y_data))

    model = gaussian_model(kernel, x_data, y_data_arr)

    data_plot()

print("minimum found : %f", min(y_data))