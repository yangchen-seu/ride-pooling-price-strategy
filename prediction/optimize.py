from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
import pickle
import predict
import numpy as np

pre = predict.Predict()


with open("tmp/OD.pickle", 'rb') as f:
    OD_dict: dict = pickle.load(f)
# for index, OD in OD_dict.items():
#     OD_dict[index][4] = 0.5
# platform_profit,lambda_w_dic = pre.run(OD_dict)

# 决策变量维度和取值范围设置
num_dimensions = len(OD_dict)
platform_profit_lis = []
# lambda_w_dic_lis = []
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
def objective_function (price_rate_list):
    for index, OD in OD_dict.items():
        OD_dict[index][4] = price_rate_list[index]
    platform_profit,lambda_w_dic = pre.run(OD_dict)
    return -platform_profit

price_rate_ranges = [Real(0.5, 1.0) for _ in range(num_dimensions)]
# 贝叶斯优化过程
result = gp_minimize(objective_function, price_rate_ranges, n_calls=100, random_state=0)

#最优解，最优目标函数值，和每次算法迭代的目标函数值
best_solution = result.x
best_function_value = result.fun
func_vals = result.func_vals

print("Best parameters: {}".format(best_solution))
print("Best function value: {}".format(best_function_value))

# 绘制寻解曲线
plt.plot(func_vals)
plt.plot(platform_profit_lis)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Optimization Progress')
plt.show()
