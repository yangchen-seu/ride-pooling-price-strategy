'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-22 04:02:38
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-12-26 09:55:43
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
        self.simu.cfg.OD_dict = OD_dict
        while True:
            OD_dict, ride_step, shared_step, done = self.simu.step()
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
        return OD_dict ,ride_step, shared_step, res 


    def main(self):


        import time
        start = time.time()

        
        iter_num = 0
        error = 1e6
        outer_all_steps = []
        while iter_num < 5:
            print('iter_num:',iter_num,'error',error)
            iter_num += 1
            OD_dict = self.cfg.OD_dict
            self.cfg.OD_dict ,ride_step, shared_step, res  = self.run(OD_dict) 
            print('self.cfg.OD_dict[0][6]',self.cfg.OD_dict[0][6])
            outer_all_steps.append([np.max(ride_step), np.max(shared_step)])
            error = np.max(outer_all_steps[len(outer_all_steps) - 1])
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

ma = Main()
ma.main()
