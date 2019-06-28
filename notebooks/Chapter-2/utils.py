import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict



# 02-XOR&Perceptron
def draw_function(func, return_fig=False):
    xx = torch.linspace(-5, 5, steps=1000)
    fig = plt.figure()
    plt.plot(xx.numpy(), func(xx).numpy())
    plt.xlabel("x", fontdict={"fontsize":16})
    plt.ylabel("y", fontdict={"fontsize":16}, rotation=0)
    plt.title(f"{func.__name__}", fontdict={"fontsize":20})
    plt.show()
    if return_fig:
        return fig
    
def check_gate(gate_func, **kwargs):
    xx = [(0, 0), (1, 0), (0, 1), (1, 1)]
    predict = torch.stack([gate_func(i, j, w=kwargs['w'], b=kwargs['b']) for i, j in xx])
    if (gate_func.__name__ == "AND"):
        target = torch.ByteTensor([0, 0, 0, 1])
    elif (gate_func.__name__ == "NAND"):
        target = torch.ByteTensor([1, 1, 1, 0])
    elif (gate_func.__name__ == "OR"):
        target = torch.ByteTensor([0, 1, 1, 1])
    elif (gate_func.__name__ == "XOR"):
        target = torch.ByteTensor([0, 1, 1, 0])
    else:
        return "gate_func error"
    for i, j in xx:
        print(f"x1={i}, x2={j}, y={gate_func(i, j, w=kwargs['w'], b=kwargs['b'])}")
    print(f"{predict.eq(target).float().sum()/len(target.float())*100} % right!")


def plot_dots(ax, gate_func):
    x = [(0, 0), (1, 0), (0, 1), (1, 1)]
    if (gate_func.__name__ == "AND"):
        marker_o = list(zip(*x[3:]))
        marker_x = list(zip(*x[:3]))
    elif (gate_func.__name__ == "NAND"):
        marker_o = list(zip(*x[:3]))
        marker_x = list(zip(*x[3:]))
    elif (gate_func.__name__ == "OR"):
        marker_o = list(zip(*x[1:]))
        marker_x = list(zip(*x[:1]))
    elif (gate_func.__name__ == "XOR"):
        marker_o = list(zip(*x[1:3]))
        marker_x = list(zip(*x[::3]))
    else:
        return "gate_func error"
    
    ax.scatter(marker_o[0], marker_o[1], c='r', marker='o', label='1')
    ax.scatter(marker_x[0], marker_x[1], c='b', marker='x', label='0')
    ax.legend()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_title(gate_func.__name__)
    ax.grid()
 
def plot_line(**kwargs):
    """x2 = (-w1*x1 - b) / w2"""
    x1 = [-2, 2]
    w = kwargs['w']
    b = kwargs['b']
    get_x2 = lambda x: (-w[0]*x - b) / w[1]
    # plot
    ax=kwargs['ax']
    ax.plot(x1, [get_x2(x1[0]), get_x2(x1[1])], c='g')

def draw_solution(x, w, b, ax, func):
    s = func(x, w=w, b=b).item()
    marker_shape = 'o' if s == 1 else 'x'
    marker_color = 'r' if s == 1 else 'b'
    ax.scatter(x.numpy()[0], x.numpy()[1], c=marker_color, marker=marker_shape, label='{}: {}'.format(func.__name__, s))
    plot_line(ax=ax, w=w, b=b)
    ax.legend()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.grid()
    ax.set_title('[{}] input: {} > result: {}'.format(func.__name__, x.long().numpy(), s))

def draw_solution_by_step(x, **kwargs):
    NAND = kwargs['f_nand']
    OR = kwargs['f_or']
    AND = kwargs['f_and']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
    s = torch.FloatTensor([NAND(x, w=kwargs['w_nand'], b=kwargs['b_nand']), 
                           OR(x, w=kwargs['w_or'], b=kwargs['b_or'])])
    draw_solution(x, w=kwargs['w_nand'], b=kwargs['b_nand'], ax=axes[0], func=NAND)
    draw_solution(x, w=kwargs['w_or'], b=kwargs['b_or'], ax=axes[1], func=OR)
    draw_solution(s, w=kwargs['w_and'], b=kwargs['b_and'], ax=axes[2], func=AND)
              
# 06-summary
from matplotlib import animation

def draw_gradient_ani(fowrard_function, his, n_step, interval, title):
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = fowrard_function(X, Y)
    levels = [0.1, 1, 2, 3, 4, 6, 8, 10, 14, 18, 24, 30, 50]
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    scat = ax.scatter([], [], s=20, c='g', edgecolors='k')
    data = np.c_[his['x'], his['y']]

    def ani_init():
        scat.set_offsets([])
        return scat,

    def ani_update(i, data, scat):
        scat.set_offsets(data[i])
        return scat,

    plt.contour(X, Y, Z, cmap='RdBu_r', levels=levels)
    plt.title(title, fontsize=20)
    plt.xlabel('x')
    plt.ylabel('y')
    anim = animation.FuncAnimation(fig, ani_update, init_func=ani_init, frames=n_step, 
                                   interval=interval, blit=True, fargs = (data, scat))
    plt.close()
    return anim

def draw_gradient_plot(fowrard_function, his, title="",  rt_fig=False):
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = fowrard_function(X, Y)
    levels = [0.1, 1, 2, 3, 4, 6, 8, 10, 14, 18, 24, 30, 50]
    fig = plt.figure(figsize=(6, 6), dpi=100)
    
    plt.contour(X, Y, Z, cmap='RdBu_r', levels=levels)
    if his is not None:
        plt.plot(his['x'], his['y'], 'go-', markersize=5, linewidth=0.5)
    plt.title(title, fontsize=20)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    if rt_fig:
        return fig

def params_init(a=-7.0, b=2.0):
    x = nn.Parameter(torch.FloatTensor([a]))
    y = nn.Parameter(torch.FloatTensor([b]))
    params = nn.ParameterDict(parameters={"x": x, "y": y})
    return params

def optim_his_init(a, b):
    params = params_init(a, b)
    optim_his = defaultdict(list)
    optim_his['x'].append(params.x.item())
    optim_his['y'].append(params.y.item())
    return optim_his, params

def simulation(fowrard_function, params, optimizer, optim_his, n_step):
    for step in range(n_step):
        optimizer.zero_grad()
        loss = fowrard_function(params.x, params.y)
        loss.backward()
        optimizer.step()
        optim_his['x'].append(params.x.item())
        optim_his['y'].append(params.y.item())

    return optim_his