import numpy as np 
import panda as pd 

def create_decay_shedule(decay_period: int, decay_func):
    x = np.arange(1, decay_period+1)
    x = decay_func(x)
    x = np.append(1,x)
    return x

def decay_func(x):
    return weibull(x, beta=.5, gamma=.5, eta=.5)

def create_ad_stock(x, decay_func):
    arr_length = len(x)
    adstock = np.zeros(arr_length,)
    decay_rate = create_decay_shedule(arr_length, decay_func)

    for i in range(arr_length):
    # print(array_shift(x,i)*decay_rate[i])
        adstock += array_shift(x,i)*decay_rate[i]
    return adstock


def geo_decay(x, theta):
    n = len(x)
    y = [1]
    for i in range(n):
        y.append(y[-1]*theta)
    return np.array(y[1::])


def decay_func_2(x):
    return geo_decay(x, 0.3)
