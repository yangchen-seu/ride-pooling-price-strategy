'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-07-03 09:04:17
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-10 03:16:08
FilePath: /matching/reinforcement learning/Seeker.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import time
import numpy as np

class Seeker():

    def __init__(self, row ) -> None:

        self.id = row['dwv_order_make_haikou_1.order_id']
        self.begin_time = row['dwv_order_make_haikou_1.departure_time']
        self.O_location = row.O_location
        self.D_location = row.D_location
        self.begin_time_stamp = time.mktime(time.strptime\
            (self.begin_time, "%Y-%m-%d %H:%M:%S"))
        self.matching_prob = row.matching_prob
        self.rs = row.ride_distance
        self.detour_distance = row.detour_distance
        self.es = row.shared_distance
        self.rst = row.ride_distance_for_taker
        self.detour_distance_for_taker = row.detour_distance_for_taker
        self.est = row.shared_distance_for_taker
        self.service_target = 0
        self.detour = 0
        self.shortest_distance = 0
        self.traveltime = 0
        self.waitingtime = 0
        self.delay = 1
        self.response_target = 0
        self.k = 1
        self.value = 0
        self.shared_distance = 0
        self.ride_distance = 0
        self.esds = row.saved_distance_for_seeker
        self.esdt = row.saved_distance_for_taker
        self.carpool_target = 0 # 是否会选择拼车
        self.od_id = 0
        self.pickup_time = row.pickup_time

    def show(self):
        print('self.id', self.id)
        print('self.begin_time',  self.begin_time)
        print('self.O_location', self.O_location)
        print('self.D_location',  self.D_location)
        print('self.begin_time_stamp', self.begin_time_stamp)
        print('self.matching_prob',  self.matching_prob)
        print('self.rs ', self.rs)
        print('self.detour_distance ', self.detour_distance)
        print('self.es ', self.es)
        print('self.rst ', self.rst)
        print('self.detour_distance_for_taker ', self.detour_distance_for_taker)
        print('self.est ', self.est)
        print('self.detour', self.detour)
        print('self.shortest_distance', self.shortest_distance)
        print('self.traveltime ', self.traveltime )
        print('self.waitingtime ', self.waitingtime )
        print('self.delay ', self.delay )
        print('self.k ', self.k )
        print('self.value ', self.value )
        print('self.shared_distance ', self.shared_distance)
        print('self.ride_distance ', self.ride_distance)    

    def set_delay(self, time):
        self.k = np.floor((time - self.begin_time_stamp) / 60 )
        self.delay = 1.1 ** self.k

    def set_value(self,value):
        self.value = value
        
    def set_shortest_path(self,distance):
        self.shortest_distance = distance

    def set_waitingtime(self, waitingtime):
        self.waitingtime = waitingtime

    def set_pickuptime(self, pickup_time):
        self.pickup_time = pickup_time

    def set_traveltime(self,traveltime):
        self.traveltime = traveltime

    def set_detour(self,detour):
        self.detour = detour

    def cal_expected_ride_distance_for_wait(self):
        self.shared_distance  =self.shared_distance 

    def set_probability(self, OD_dict):
        theta = OD_dict[4]
        A = 1 # 2.05
        c_0 = 3
        a = 1
        beta = 2
        l = self.shortest_distance / 1000 
        t_p = 2  # if self.waitingtime == 0 else self.waitingtime / 60
        T_p = OD_dict[5] / 600 # 用时多少min
        t_s = t_p + 0.5
        T_s = self.shortest_distance  / 600 # 用时多少m
        # print('tmp',(A * ((beta* (t_p + T_p) + theta * l * c_0 +a) - (beta * (t_s + T_s) + l * c_0))))
        # print('theta',theta)
        return 1 / (1 + np.exp(A * ((beta* (t_p + T_p) + theta * l * c_0 +a) - (beta * (t_s + T_s) + l * c_0))))
    
    def set_discount(self,discount):
        self.discount = discount

    def set_od_id(self, od_id):
        self.od_id = od_id