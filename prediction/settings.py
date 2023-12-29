'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-20 03:22:31
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-12-20 08:23:25
FilePath: /yangchen/ridepooling-pricing/ride-pooling-price-strategy/prediction/settings.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
配置文件
Sun Nov 28 2021
Copyright (c) 2021 Yuzhen FENG
"""
params = {
    'lowest_road_class': 5,
    'max_combined_length': 1000,
    'OD_num': 1698,
    'OD_file':'OD.csv',
    'process_num': 4,
    'chunk_num': 5,
    'search_radius': 2000,
    'max_detour': 3000,
    'w_detour': 0,
    'w_pickup': 0,
    'w_shared': 0,
    "w_ride": -1,
    'pickup_time': 2,
    'speed': 600, # m/min
    'max_iter_time': 200,
    "min_iter_time": 5,
    'convergent_condition': 1e-6,
    'outer_max_iter_time': 100,
    'outer_convergent_condition': 1e-2,
    'M': 1e6,
    'epsilon': 1e-6,
    'n_v':300,
    'beta':1,
    'delta':1,
}
