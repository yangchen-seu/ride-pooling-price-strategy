'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-22 04:02:38
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-12 09:05:45
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

import pickle

class Main():

    def __init__(self, cfg, discount = 0.8, ) -> None:
        self.cfg = cfg
        self.simu = sm.Simulation(self.cfg)
        self.test_episodes = 1
        OD_data = pd.read_csv('input/OD.csv')
        OD_num = OD_data.shape[0]  # OD数
        OD_dict = dict()
        self.cfg.OD_dict = OD_dict
        self.cfg.method = 'uniformed'
        order = pd.read_csv('input/predict_result.csv')
        
        for i in range(OD_num):
            df = order[order['OD_id'] == int(OD_data.loc[i].loc["OD_id"])]
            ride_distance = df['ride_distance'].mean()
            shared_distance = df['shared_distance'].mean()
            self.cfg.OD_dict[int(OD_data.loc[i].loc["OD_id"])] = [int(OD_data.loc[i].loc["origin_id"]),
                                                        int(OD_data.loc[i].loc["destination_id"]),
                                                        OD_data.loc[i].loc["lambda"],
                                                        OD_data.loc[i].loc["solo_distance"],
                                                        discount,
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

        return OD_dict ,ride_step, shared_step, res 


    def main(self):


        import time
        start = time.time()

        
        iter_num = 0
        error = 1e6
        outer_all_steps = []
        while iter_num < 20 and error > 10 : # or iter_num < 5
        # while iter_num < 20:
            iter_num += 1
            OD_dict = self.cfg.OD_dict
            self.cfg.OD_dict ,ride_step, shared_step, res  = self.run(OD_dict) 
            outer_all_steps.append([np.max(ride_step), np.max(shared_step)])
            error = np.max(outer_all_steps[len(outer_all_steps) - 1])
            print('iter_num:',iter_num,'error',error, 'platform_income',res['platform_income'])
        end = time.time()
        print('执行时间{}'.format(end - start))
        print(res)
        return res

def different_ratio():
    ratios = [100/25,100/50,100/75]
    for ratio in ratios:
        cfg = Config.Config()
        cfg.order_driver_ratio = ratio
        cfg.driver_ratio_target = True
        cfg.progress_target = False
        print('ratio:',cfg.order_driver_ratio)

        ma = Main(cfg, discount = 0.8)
        ma.cfg.unit_distance_cost = 2.2
        ma.main()

cfg = Config.Config()
ma = Main(cfg, 0.8)
ma.cfg.unit_distance_cost = 2.2
ma.main()
    
# for discount in [0.6,0.7,0.8,0.85,0.875,0.9,0.95]:
#     print('discount',discount)
#     cfg = Config.Config()
#     ma = Main(cfg, discount)
#     ma.cfg.unit_distance_cost = 2.2
#     ma.main()


# lis = {}

# for i in range(50):
#     print(i)
#     iter_num = 0
#   cfg = Config.Config()
#     ma = Main(cfg, 0.8)
#     res = ma.main()
#     lis[i] = res
#     print(res)
#     # Correct the file handling for pickle
# with open('./result/uniformed_randomness.pickle', 'wb') as file:
#     pickle.dump(lis, file)


        
# different_ratio()