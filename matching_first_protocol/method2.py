'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-22 04:02:38
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-12-25 09:51:23
FilePath: /yangchen/ridepooling-pricing/ride-pooling-price-strategy/matching_first_protocol/uniformed.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import Simulation as sm
import Config
import pandas as pd


class Main():

    def __init__(self,cfg ) -> None:
        self.simu = sm.Simulation(cfg)
        self.test_episodes = 1
 
 
    def run(self):
        while True:
            reward, done = self.simu.step()
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
        
        return res


def one_day_test():
    cfg = Config.Config()
    OD_data = pd.read_csv('input/OD.csv')
    OD_num = OD_data.shape[0]  # OD数
    OD_dict = dict()
    cfg.OD_dict = OD_dict
    for i in range(OD_num):
        OD_dict[int(OD_data.loc[i].loc["OD_id"])] = [int(OD_data.loc[i].loc["origin_id"]),
                                                    int(OD_data.loc[i].loc["destination_id"]),
                                                    OD_data.loc[i].loc["lambda"],
                                                    OD_data.loc[i].loc["solo_distance"],
                                                    OD_data.loc[i].loc["theta"],]  # 列表[起点id，终点id，lambda]

    import time
    start = time.time()
    ma = Main(cfg)
    ma.run() 
    end = time.time()
    print('执行时间{}'.format(end - start))

one_day_test()
