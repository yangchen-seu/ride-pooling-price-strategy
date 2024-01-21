'''
Author: your name
Date: 2021-12-07 21:36:06
LastEditTime: 2024-01-15 17:20:19
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \强化学习网约车\Reinforcementlearning\Config.py
'''

class Config:

    def __init__(self) -> None:
        self.device = 'cpu'
        # files
        self.input_path = 'input\\'
        self.output_path = 'output\\'
        
        self.order_file= './input/orders_predict.csv'  # order
        self.network_file = 'network.csv'
        self.shortest_path_file = './input/shortest_path.csv'

        # 与网络相关的参数
        self.unit_distance_value = 5 # 平台收益一公里5块钱
        self.unit_distance_cost = 2 # 司机消耗一公里2块钱
        
        self.date = '2017-05-02'
        self.simulation_begin_time = ' 05:00:00' # 仿真开始时间
        self.simulation_end_time = ' 06:00:00' # 仿真结束时间
        self.unit_driving_time = 120/1000 # 行驶速度
        self.unit_time_value = 1.5/120 # 每秒的行驶费用

        self.driver_num = 300
        self.order_driver_ratio = 100 / 25
        self.driver_ratio_target = False
        # self.progress_target =  True #  False # True
        self.progress_target =  False # True

        self.optimazition_target = 'expected_shared_distance' # platform_income, expected_shared_distance
        self.matching_condition = True
        # matching condition
        self.pickup_distance_threshold = 2000
        self.detour_distance_threshold = 3000
        self.delay_time_threshold = 300
        self.dead_value = -1e4
        self.reposition_time_threshold = 120

        self.OD_dict = {}
        
        self.method = 'default'
        # self.randomness = False
        self.randomness = True