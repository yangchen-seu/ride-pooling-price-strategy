'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-22 04:02:38
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-15 17:19:32
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
            reward, ride_step, shared_step, done = self.simu.step()
            if done:
                break

        res = self.simu.res        
        return res


def one_day_test():
    cfg = Config.Config()
    # cfg.randomness = True
    OD_data = pd.read_csv('input/OD.csv')
    OD_num = OD_data.shape[0]  # OD数
    OD_dict = dict()
    cfg.OD_dict = OD_dict
    cfg.method = 'method2'
    order = pd.read_csv('input/predict_result.csv')
    
    
    for i in range(OD_num):
        df = order[order['OD_id'] == int(OD_data.loc[i].loc["OD_id"])]
        ride_distance = df['ride_distance'].mean()
        shared_distance = df['shared_distance'].mean()
        OD_dict[int(OD_data.loc[i].loc["OD_id"])] = [int(OD_data.loc[i].loc["origin_id"]),
                                                    int(OD_data.loc[i].loc["destination_id"]),
                                                    OD_data.loc[i].loc["lambda"],
                                                    OD_data.loc[i].loc["solo_distance"],
                                                    OD_data.loc[i].loc["theta"],
                                                    ride_distance,
                                                        shared_distance,]  # 列表[起点id，终点id，lambda]

    import time
    start = time.time()
    ma = Main(cfg)
    res = ma.run() 
    print(res)
    end = time.time()
    print('执行时间{}'.format(end - start))
    return res
    
import pickle 
def randomness_test():
    lis = {}
    for i in range(50):
        print(i)
        lis [i] = one_day_test()
        print(lis[i])
    with open('./result/method2_randomness.pickle', 'wb') as file:
        pickle.dump(lis, file)
        
        
        
def different_ratio():
    ratios = [100/25,100/50,100/75]
    for ratio in ratios:
        cfg = Config.Config()
        cfg.order_driver_ratio = ratio
        cfg.driver_ratio_target = True
        cfg.progress_target = False
        print('ratio:',cfg.order_driver_ratio)

        OD_data = pd.read_csv('input/OD.csv')
        OD_num = OD_data.shape[0]  # OD数
        OD_dict = dict()
        cfg.OD_dict = OD_dict
        cfg.method = 'method2'
        order = pd.read_csv('input/predict_result.csv')
        
        
        for i in range(OD_num):
            df = order[order['OD_id'] == int(OD_data.loc[i].loc["OD_id"])]
            ride_distance = df['ride_distance'].mean()
            shared_distance = df['shared_distance'].mean()
            OD_dict[int(OD_data.loc[i].loc["OD_id"])] = [int(OD_data.loc[i].loc["origin_id"]),
                                                        int(OD_data.loc[i].loc["destination_id"]),
                                                        OD_data.loc[i].loc["lambda"],
                                                        OD_data.loc[i].loc["solo_distance"],
                                                        OD_data.loc[i].loc["theta"],
                                                        ride_distance,
                                                            shared_distance,]  # 列表[起点id，终点id，lambda]

        import time
        start = time.time()
        ma = Main(cfg)
        res = ma.run() 
        print(res)
        end = time.time()
        print('执行时间{}'.format(end - start))
        
# one_day_test()
randomness_test()
# different_ratio()