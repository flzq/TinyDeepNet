import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
# from mnist import load_mnist


def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1;

def print_res(x1, x2, fun):
    print(fun(x1, x2))

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# 一维数组版本：一个样本
# def softmax(x):
#     c = np.max(x)
#     exp_a = np.exp(x - c)  # 防止溢出
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#
#     return y

# 多维数组版本：多个样本
def softmax(x):
    # 对于多维数组
    if x.ndim == 2:
        x = x.T
        c = np.max(x, axis=0)
        exp_a = np.exp(x - c)
        sum_exp_a = np.sum(exp_a, axis=0)
        y = exp_a / sum_exp_a
        return y.T

    # 对于一维数组
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# 均方误差函数
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 交叉熵误差
def cross_entropy_error(y, t):
    if y.ndim == 1:  # 处理单个数据
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    delta = 1e-7  # 防止计算 np.log 时溢出
    return -np.sum(t * np.log(y + delta)) / batch_size

# 数值微分：利用微小的差分求导数的过程
# 采用中心差分的方式计算
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

# 梯度：一维数组
# def numerical_gradient(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x)
#
#     for idx in range(x.size):
#         tmp_val = x[idx]
#         # f(x+h)的计算
#         x[idx] = tmp_val + h
#         fxh1 = f(x)
#         # f(x-h)的计算
#         x[idx] = tmp_val - h
#         fxh2 = f(x)
#         grad[idx] = (fxh1 - fxh2) / (2 * h)
#         x[idx] = tmp_val
#
#     return grad

# 梯度：多维数组
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    # nditer迭代对象，it用于访问数组的每个元素
    # op_flags=['readwrite']：迭代器可以对数组进行读写操作
    # flags=['multi_index']：用多索引访问x、grad数组的对应位置
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # 获得当前位置索引
        idx = it.multi_index
        tmp_val = x[idx]
        # f(x+h) 计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h) 计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        # 计算当前位置idx的梯度
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        # 还原值
        x[idx] = tmp_val
        # it指向下一个位置
        it.iternext()

    return grad

# 梯度下降法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

def fun1(x):
    return 0.01*x**2 + 0.1*x

def fun2(x):
    return x[0]**2 + x[1]**2

def fun3(w):
    x = np.array([0.6, 0.9])
    x = np.dot(x, w)
    x = softmax(x)
    loss = cross_entropy_error(x, np.array([0, 0, 1]))
    return loss

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, X):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(X, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3

    return a3

class MulLayer(object):
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

def mullayer():
    apple = 100
    apple_num = 2
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # forward
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)
    print(price)

    # backward
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    print(dapple, dapple_num, dapple_price, dtax)

class AddLayer(object):
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

def apple_orange():
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_price_layer = AddLayer()
    mul_tax_layer = MulLayer()

    # forward
    apple_price = mul_apple_layer.forward(apple_num, apple)
    orange_price = mul_orange_layer.forward(orange_num, orange)
    add_price = add_price_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(add_price, tax)
    print(apple_price, orange_price, add_price, price)

    # backward
    dprice = 1
    dadd_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_price_layer.backward(dadd_price)
    dapple_num, dapple = mul_apple_layer.backward(dapple_price)
    dorange_num, dorange = mul_orange_layer.backward(dorange_price)
    print(dadd_price, dtax)
    print(dapple_price, dorange_price)
    print(dapple_num, dapple)
    print(dorange_num, dorange)

if __name__ == '__main__':
    apple_orange()