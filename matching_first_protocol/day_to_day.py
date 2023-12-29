'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-22 04:02:38
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-12-25 08:22:48
FilePath: /yangchen/ridepooling-pricing/ride-pooling-price-strategy/matching_first_protocol/uniformed.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE

OD_dict
0:origin_id
1:destination_id
2:lambda
3:solo_distance
4:theta
5:ride_distance
6:shared_distance

'''

import Simulation as sm
import Config
import pandas as pd
import numpy as np
from scipy.optimize import minimize



class Main():

    def __init__(self, ) -> None:
        self.cfg = Config.Config()
        self.simu = sm.Simulation(self.cfg)
        self.test_episodes = 1
        OD_data = pd.read_csv('input/OD.csv')
        OD_num = OD_data.shape[0]  # OD数
        OD_dict = dict()
        self.cfg.OD_dict = OD_dict
        order = pd.read_csv('input/predict_result.csv')
        
        for i in range(OD_num):
            df = order[order['OD_id'] == int(OD_data.loc[i].loc["OD_id"])]
            ride_distance = df['ride_distance'].mean()
            shared_distance = df['shared_distance'].mean()
            self.cfg.OD_dict[int(OD_data.loc[i].loc["OD_id"])] = [int(OD_data.loc[i].loc["origin_id"]),
                                                        int(OD_data.loc[i].loc["destination_id"]),
                                                        OD_data.loc[i].loc["lambda"],
                                                        OD_data.loc[i].loc["solo_distance"],
                                                        0.8,
                                                        ride_distance,
                                                        shared_distance,]  # 列表[起点id，终点id，lambda] 
 

    def run(self, OD_dict):
        self.simu.reset()
        self.cfg.OD_dict = OD_dict
        while True:
            OD_dict,  done = self.simu.step()
            if done:
                break

        res = self.simu.res
        print('waitingTime:',res['waitingTime'])
        print('detour_distance:',res['detour_distance'])
        print('pickup_time:',res['pickup_time'])
        print('shared_distance:',res['shared_distance'])
        print('total_ride_distance:',res['total_ride_distance'])
        print('saved_ride_distance:',res['saved_ride_distance'])
        print('platform_income:',res['platform_income'])
        print('response_rate:',res['response_rate'])
        print('carpool_rate:',res['carpool_rate'])
        return OD_dict , res 


    def main(self):


        import time
        start = time.time()

        
        iter_num = 0
        error = 1e6
        while iter_num < 200 and error > 1e-2 or iter_num < 5:
            print('iter_num:',iter_num,'error',error)
            iter_num += 1
            OD_dict = self.cfg.OD_dict
            error, OD_dict = self.day_to_day(OD_dict)
            self.cfg.OD_dict , res  = self.run(OD_dict) 
        end = time.time()
        print('执行时间{}'.format(end - start))
        print('waitingTime:',res['waitingTime'])
        print('detour_distance:',res['detour_distance'])
        print('pickup_time:',res['pickup_time'])
        print('shared_distance:',res['shared_distance'])
        print('total_ride_distance:',res['total_ride_distance'])
        print('saved_ride_distance:',res['saved_ride_distance'])
        print('platform_income:',res['platform_income'])
        print('response_rate:',res['response_rate'])
        print('carpool_rate:',res['carpool_rate'])

    def day_to_day(self,  OD_dict):
        
        theta_step = []
        outer_all_steps = []
        alpha = 0.1
        # 不动点更新
        for od_id in OD_dict.keys():
            # update            
            def fw( OD_dict, theta):
                A = 1
                c_0 = 3
                a = 1
                beta = 2
                l = OD_dict[3] / 1000 
                t_p = 2
                T_p = OD_dict[5] / 600 # 用时多少min
                t_s = 2.5
                T_s = OD_dict[3]  / 600 # 用时多少m
                res = 1 / (1 + np.exp(A * ((beta* (t_p + T_p) + theta * l * c_0 +a) - (beta * (t_s + T_s) + l * c_0))))
                # print('res',res,'theta',theta)
                return res

            # 定义目标函数
            def objective_function(x, *args):
                OD_dict = args[0]  # 提取参数
                # profit
                profit = (x * OD_dict[3] / 1000 * 2.5  - 2 * (OD_dict[5] - OD_dict[6] /2 )  /1000) * fw(OD_dict, x)
                # print('OD_dict[3],{},OD_dict[5],{},OD_dict[6],{},profit{}'.format(OD_dict[3],OD_dict[5],OD_dict[6],profit))
                return -profit

            # profit = []
            # fw_lis = []
            # import matplotlib.pyplot as plt
            # for x in np.arange(0, 1.05, 0.05):
            #     profit.append (objective_function(x, (OD_dict[od_id])))
            #     fw_lis.append (fw(OD_dict[od_id], x))
            # plt.plot(np.arange(0, 1.05, 0.05),profit)
            # plt.plot(np.arange(0, 1.05, 0.05),fw_lis)
            # plt.savefig('output/test.png')
            # exit()
            # 使用 minimize 函数求解极小值点
            # print('init x0',OD_dict[od_id][4])
            result = minimize(objective_function, x0= OD_dict[od_id][4], args=(OD_dict[od_id],), bounds=[(0.7,0.9)]) # 

            # 输出结果
            # print("Optimal solution:", result.x)
            # print("Optimal function value:", result.fun)

            theta_dic_tmp = OD_dict[od_id][4] # seekers[od_id]['lambda']
            OD_dict[od_id][4] = theta_dic_tmp  * alpha + (1-alpha ) * result.x
            # print('f_w_function(C_w_dic[od_id] )',f_w_function(C_w_dic[od_id]['pool'] ,  C_w_dic[od_id]['solo']))
            theta_step.append(abs(OD_dict[od_id][4] - theta_dic_tmp))
        outer_all_steps.append([np.max(theta_step)])
        outer_error = np.max(outer_all_steps[len(outer_all_steps) - 1])
    
        return outer_error, OD_dict

ma = Main()
ma.main()
