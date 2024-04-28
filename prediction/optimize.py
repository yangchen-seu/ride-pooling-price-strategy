'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-20 03:22:31
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-09 04:00:36
FilePath: /yangchen/ridepooling-pricing/ride-pooling-price-strategy/prediction/optimize.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
import pickle
import predict
import numpy as np
import time
import matplotlib
from zoopt import Dimension, ValueType, Dimension2, Objective, Parameter, ExpOpt,Opt

matplotlib.use('Agg')  # 选择一个适合远程环境的后端，如Agg




pre = predict.Predict()

begin = time.time()

with open("tmp/OD.pickle", 'rb') as f:
    OD_dict: dict = pickle.load(f)
# for index, OD in OD_dict.items():
#     OD_dict[index][4] = 0.5
# platform_profit,lambda_w_dic = pre.run(OD_dict)

# 决策变量维度和取值范围设置
num_dimensions = len(OD_dict)
platform_profit_lis = []
lambda_w_dic_lis = []
# for rate in np.arange(0.1, 1.1, 0.1).tolist():
# for rate in [0.7,0.8,0.9]:
#     print('***'*6,rate,'***'*6,)
#     price_rate_ranges = [rate for _ in range(num_dimensions)]
#     for index, OD in OD_dict.items():
#         OD_dict[index][4] = price_rate_ranges[index]
#     platform_profit,lambda_w_dic = pre.run(OD_dict)
#     platform_profit_lis.append (platform_profit)
#     lambda_w_dic_lis.append(lambda_w_dic.values())
# for i in range(len(lambda_w_dic_lis) - 1):
#     # 计算两个列表对应位置元素的差值
#     differences = [abs(x - y) for x, y in zip(lambda_w_dic_lis[i], lambda_w_dic_lis[i+1])]

#     # 计算差值的平均值或其他统计量
#     average_difference = sum(differences) / len(differences)
#     print(average_difference)
# print('platform_profit_lis',platform_profit_lis)


# def objective_function (price_rate_list):
#     for index, OD in OD_dict.items():
#         OD_dict[index][4] = price_rate_list[index]
#     platform_profit,lambda_w_dic = pre.run(OD_dict)
#     print('platform_profit',platform_profit)
#     return -platform_profit

# price_rate_ranges = [Real(0.7, 0.9) for _ in range(num_dimensions)]
# # 贝叶斯优化过程
# result = gp_minimize(objective_function, price_rate_ranges, acq_optimizer = 'lbfgs', n_calls=num_dimensions * 100, random_state=0, n_jobs=-1)

# #最优解，最优目标函数值，和每次算法迭代的目标函数值
# best_solution = result.x
# best_function_value = result.fun
# func_vals = result.func_vals
# end = time.time()
# print('execute time',end - begin)
# print("Best parameters: {}".format(best_solution))
# print("Best function value: {}".format(best_function_value))
# # 绘制寻解曲线
# plt.plot(platform_profit_lis)
# plt.plot(func_vals)
# plt.xlabel('Iteration')
# plt.ylabel('Objective Function Value')
# plt.title('Optimization Progress')
# plt.show()
# plt.savefig('result/BO_iteration.png')
# plt.close()


# SARCO
def objective_function(solution):
    # 获取 Solution 的维度值
    price_rate_list = solution.get_x()
    for index, OD in OD_dict.items():
        OD_dict[index][4] = price_rate_list[index]
    
    platform_profit, lambda_w_dic = pre.run(OD_dict)
    platform_profit_lis.append(platform_profit)
    print('profit',platform_profit)
    return -platform_profit


# 决策变量维度，取值范围等基本设置

dim_regs = [[0.7,0.9]] * num_dimensions  # dimension range
dim_tys = [True] * num_dimensions  # dimension type : real
dim = Dimension(num_dimensions, dim_regs, dim_tys)  # form up the dimension object
objective = Objective(objective_function, dim)  # form up the objective function

# parallel optimization for time-consuming tasks
solution_opt = Opt.min(objective, Parameter(budget=  100 * num_dimensions, parallel=True, server_num=128))

#最优解，最优目标函数值，和每次算法迭代的目标函数值
best_solution = solution_opt.get_x()
best_function_value = solution_opt.get_value()
history_value = objective.get_history_bestsofar()

print("Best parameters: {}".format(best_solution))
print("Best function value: {}".format(best_function_value))

print('profit',platform_profit_lis)

# 绘制寻解曲线
plt.plot(platform_profit_lis, label='Platform Profit')
plt.plot(history_value, label='Optimization Progress')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Optimization Progress')
plt.legend()  # 添加图例
plt.savefig('result/SRACO_iteration.png')
plt.close()






'''  HEBO
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

# Define the design space
params = [
    {'name': str(i), 'type': 'num', 'lb': 0.7, 'ub': 0.9} for i in range(num_dimensions)
]


design_space = DesignSpace().parse(params)

# Define the objective function
def objective_function(solution):
    # 获取 Solution 的维度值
    price_rate_list = solution
    for index, OD in OD_dict.items():
        OD_dict[index][4] = price_rate_list[str(index)].values[0]

    
    platform_profit, lambda_w_dic = pre.run(OD_dict)
    platform_profit_lis.append(platform_profit)
    print('profit',platform_profit)
    return -platform_profit


hebo_batch = HEBO(design_space, model_name = 'gpy')
for i in range(100):
    rec_x = hebo_batch.suggest(n_suggestions=1)
    hebo_batch.observe(rec_x, objective_function(rec_x))
    print('Iter %d, best_y = %.2f' % (i, hebo_batch.y.min()))
    

# Get the best solution and value
conv_hebo_batch = np.minimum.accumulate(hebo_batch.y)

plt.figure(figsize = (8,6))
plt.semilogy(conv_hebo_batch, 'x-',label = 'HEBO, Parallel,Batch = 128')
plt.xlabel('Evaluation')
plt.ylabel('Regret')
plt.legend()


'''