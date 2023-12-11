"""
不动点迭代过程及预测拼车概率、期望行驶里程与绕行里程
Tues Dec 7 2021
Copyright (c) 2021 Yuzhen FENG
"""
from os import error
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import time
import os
from progress.bar import Bar
from settings import params



import itertools
import os
import pickle
import networkx as nx
import pandas as pd
from multiprocessing import Pool
import time
import traceback
from settings import params

if True:
    with open("tmp/graph.pickle", 'rb') as f:
        G: nx.Graph = pickle.load(f)
    with open("tmp/ego_graph.pickle", 'rb') as f:
        subgraph = pickle.load(f)

def search_matching_OD(OD_idx1,OD2):
    """Search matching pairs whose seeker-states belong to OD1 and taker-states to OD2

    Args:
        OD_idx1 (int): The ID of OD1
        OD2 (list): The information of OD2

    Returns:
        int: destination of OD2
    """
    nearest_G: nx.Graph = subgraph[OD_idx1]

    if nearest_G.has_node(OD2[1]): # 附近有给定OD2的终点
        pickup_distance = nearest_G.nodes[OD2[1]]["weight"]
        if pickup_distance <= params["search_radius"]:
            return OD2[1]
        else:
            return False
    else:
        return False



start_time = time.time()
# -------------------- Start the fixed point iteration --------------------
# ---------- Load data ----------
with open("tmp/link.pickle", 'rb') as f:
    link_dict: dict = pickle.load(f)
with open("tmp/OD.pickle", 'rb') as f:
    OD_dict: dict = pickle.load(f)
with open("tmp/node.pickle", 'rb') as f:
    node_dict: dict = pickle.load(f)
with open("tmp/shortest_path.pickle", 'rb') as f:
    path_dict: dict = pickle.load(f)

# isPrestored = input("Do you want to use the result of the last executation? (y/n) ")
isPrestored = 'n'
if isPrestored == 'n':
    match_df = pd.read_csv("result/match.csv", index_col=["seeker_id", "taker_id", "link_idx"])
    matches = dict()
    for index, row in match_df.iterrows():
        # 将df转为字典
        matches.setdefault(index[0], []).append({"taker_id": index[1], "link_idx": index[2], "preference": row["preference"],
                                                "ride_seeker": row["ride_seeker"], "ride_taker": row["ride_taker"],
                                                "detour_seeker": row["detour_seeker"], "detour_taker": row["detour_taker"], "shared": row["shared"], "destination": row["destination"],"eta_match": 0})
    print("Data loaded.")
    # ---------- Initialize seekers and takers ----------
    seekers = dict()
    for seeker_id, OD in OD_dict.items():
        seekers[seeker_id] = {"lambda": OD[2], "p_seeker": OD[2]}

    takers = dict()
    for taker_id in OD_dict.keys():
        # if taker_id >= params["OD_num"]:
        #     continue
        takers[taker_id] = dict()
        path = path_dict[taker_id][:-1]
        takers[taker_id][0] = dict({"tau_bar": params['pickup_time'], "lambda_taker": 0, "p_taker": 0.1, "rho_taker": 0, "eta_taker": 0})
        for link_idx, link_id in enumerate(path):
            takers[taker_id][link_idx + 1] = {"tau_bar": link_dict[link_id][2] / params['speed'], "lambda_taker": 0, "p_taker": 0.1, "rho_taker": 0, "eta_taker": 0}
else:
    with open("variables/seekers.pickle", 'rb') as f:
        seekers: dict = pickle.load(f)
    with open("variables/takers.pickle", 'rb') as f:
        takers: dict = pickle.load(f)
    with open("variables/matches.pickle", 'rb') as f:
        matches: dict = pickle.load(f)
print("Variables initialized.")
# ---------- Start to iterate ----------
iter_start_time = time.time()
all_steps = []
iter_num = 0
error = params['M']
print("Outer Iterating... |", end='')
while iter_num < params['outer_max_iter_time'] and error > params['outer_convergent_condition'] or iter_num < params["min_iter_time"]:
    print("Inner Iterating... |", end='')
    lambda_w_step = []
    t_pk_w_step = []
while iter_num < params['max_iter_time'] and error > params['convergent_condition'] or iter_num < params["min_iter_time"]:
    print(iter_num % 10, end='', flush=True)
    lambda_taker_step = []
    p_seeker_step = []
    p_taker_step = []
    rho_taker_step = []
    for seeker_id, takers_of_seeker in matches.items():
        eta_match_product = seekers[seeker_id]["lambda"]
        for taker in takers_of_seeker:
            taker["eta_match"] = eta_match_product
            takers[taker["taker_id"]][taker["link_idx"]]["eta_taker"] += eta_match_product
            eta_match_product *= 1 - takers[taker["taker_id"]][taker["link_idx"]]["rho_taker"]
    
    for seeker_id in seekers.keys():
        origin_p_seeker = seekers[seeker_id]["p_seeker"]
        product = 1
        for taker in matches[seeker_id]:
            product *= 1 - takers[taker["taker_id"]][taker["link_idx"]]["rho_taker"]
        seekers[seeker_id]["p_seeker"] = 1 - product
        p_seeker_step.append(abs(seekers[seeker_id]["p_seeker"] - origin_p_seeker))
        
    for taker_id, taker in takers.items():
        for link_idx, link in taker.items():
            origin_p_taker = link["p_taker"]
            origin_rho_taker = link["rho_taker"]
            if link["eta_taker"] == 0:
                link["p_taker"] = 0
                link["rho_taker"] = link["lambda_taker"] * link["tau_bar"]
            else:
                link["p_taker"] = 1 - np.exp(-link["eta_taker"] * link["tau_bar"])
                link["rho_taker"] = link["lambda_taker"] * link["p_taker"] / link["eta_taker"]
            link["eta_taker"] = 0
            p_taker_step.append(abs(link["p_taker"] - origin_p_taker))
            rho_taker_step.append(abs(link["rho_taker"] - origin_rho_taker))

    for taker_id in takers.keys():
        for link_idx in takers[taker_id].keys():
            origin_lambda_taker = takers[taker_id][link_idx]["lambda_taker"]
            if link_idx == 0:
                takers[taker_id][link_idx]["lambda_taker"] = seekers[taker_id]["lambda"] * (1 - seekers[taker_id]["p_seeker"])
            else:
                takers[taker_id][link_idx]["lambda_taker"] = takers[taker_id][link_idx - 1]["lambda_taker"] * (1 - takers[taker_id][link_idx - 1]["p_taker"])
            lambda_taker_step.append(abs(takers[taker_id][link_idx]["lambda_taker"] - origin_lambda_taker))
    
    iter_num += 1
    if iter_num >= params["min_iter_time"]:
        all_steps.append([np.max(lambda_taker_step), np.max(p_seeker_step), np.max(p_taker_step), np.max(rho_taker_step)])
        error = np.max(all_steps[len(all_steps) - 1])
iter_end_time = time.time()
print("\nConverge! It costs:", iter_end_time - iter_start_time)
print("The average time of iteration:", (iter_end_time - iter_start_time) / iter_num)
# ---------- Plot the iteration ----------
plt.plot(np.arange(len(all_steps)) + params["min_iter_time"], all_steps, label=["lambda t(a, w)", "p s(w)", "p t(a, w)", "rho t(a, w)"])
plt.ylabel("Delta")
plt.xlabel("Iteration Time")
plt.title("Iteration")
plt.yscale("log")
plt.legend()
plt.savefig("result/iteration.png")

# ---------- Calculate the prediction result ----------
for seeker_id in seekers.keys():
    seekers[seeker_id]["matching_prob"] = 1 - takers[seeker_id][len(takers[seeker_id]) - 1]["lambda_taker"] * (1 - takers[seeker_id][len(takers[seeker_id]) - 1]["p_taker"]) / seekers[seeker_id]["lambda"]
for seeker_id, takers_of_seeker in matches.items():
    seekers[seeker_id]["total_ride_distance"] = 0
    seekers[seeker_id]["total_detour_distance"]= 0
    seekers[seeker_id]["total_shared_distance"]= 0
    seekers[seeker_id]["total_saved_distance"] = 0
    seekers[seeker_id]["total_matching_rate"] = 0
    for taker in takers_of_seeker:
        L1 = path_dict[seeker_id][-1]
        L2 = path_dict[taker["taker_id"]][-1]

        seekers[seeker_id]["total_ride_distance"] += taker["ride_seeker"] * taker["eta_match"] * takers[taker["taker_id"]][taker["link_idx"]]["rho_taker"]
        seekers[seeker_id]["total_detour_distance"] += taker["detour_seeker"] * taker["eta_match"] * takers[taker["taker_id"]][taker["link_idx"]]["rho_taker"]
        seekers[seeker_id]["total_shared_distance"] += taker["shared"] * taker["eta_match"] * takers[taker["taker_id"]][taker["link_idx"]]["rho_taker"]
        seekers[seeker_id]["total_saved_distance"] += (L1 + L2 - (taker["ride_seeker"] + taker["ride_taker"] - taker["shared"])) * taker["eta_match"] * takers[taker["taker_id"]][taker["link_idx"]]["rho_taker"]
        seekers[seeker_id]["total_matching_rate"] += taker["eta_match"] * takers[taker["taker_id"]][taker["link_idx"]]["rho_taker"]

        takers[taker["taker_id"]][taker["link_idx"]]["total_ride_distance"] = takers[taker["taker_id"]][taker["link_idx"]].setdefault("total_ride_distance", 0) + taker["ride_taker"] * taker["eta_match"] * takers[taker["taker_id"]][taker["link_idx"]]["rho_taker"]
        takers[taker["taker_id"]][taker["link_idx"]]["total_detour_distance"] = takers[taker["taker_id"]][taker["link_idx"]].setdefault("total_detour_distance", 0) + taker["detour_taker"] * taker["eta_match"] * takers[taker["taker_id"]][taker["link_idx"]]["rho_taker"]
        takers[taker["taker_id"]][taker["link_idx"]]["total_shared_distance"] = takers[taker["taker_id"]][taker["link_idx"]].setdefault("total_shared_distance", 0) + taker["shared"] * taker["eta_match"] * takers[taker["taker_id"]][taker["link_idx"]]["rho_taker"]
        takers[taker["taker_id"]][taker["link_idx"]]["total_saved_distance"] = takers[taker["taker_id"]][taker["link_idx"]].setdefault("total_saved_distance", 0) + (L1 + L2 - (taker["ride_seeker"] + taker["ride_taker"] - taker["shared"])) * taker["eta_match"] * takers[taker["taker_id"]][taker["link_idx"]]["rho_taker"]
        takers[taker["taker_id"]][taker["link_idx"]]["total_matching_rate"] = takers[taker["taker_id"]][taker["link_idx"]].setdefault("total_matching_rate", 0) + taker["eta_match"] * takers[taker["taker_id"]][taker["link_idx"]]["rho_taker"]

for seeker_id in seekers.keys():
    L = path_dict[seeker_id][-1]
    seekers[seeker_id]["ride_distance"] = 0
    seekers[seeker_id]["detour_distance"] = 0
    seekers[seeker_id]["shared_distance"] = 0
    seekers[seeker_id]["saved_distance"] = 0
    seekers[seeker_id]["ride_distance_for_taker"] = 0
    seekers[seeker_id]["detour_distance_for_taker"] = 0
    seekers[seeker_id]["shared_distance_for_taker"] = 0
    seekers[seeker_id]["saved_distance_for_taker"] = 0

    seekers[seeker_id]["ride_distance_for_seeker"] = seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_ride_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])
    seekers[seeker_id]["detour_distance_for_seeker"] = seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_detour_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])
    seekers[seeker_id]["shared_distance_for_seeker"] = seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_shared_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])
    seekers[seeker_id]["saved_distance_for_seeker"] = seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_saved_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])

    seekers[seeker_id]["ride_distance"] += seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_ride_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])
    seekers[seeker_id]["detour_distance"] += seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_detour_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])
    seekers[seeker_id]["shared_distance"] += seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_shared_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])
    seekers[seeker_id]["saved_distance"] += seekers[seeker_id]["p_seeker"] * seekers[seeker_id].setdefault("total_saved_distance", 0) / seekers[seeker_id].setdefault("total_matching_rate", params["epsilon"])

    lambda_become_taker = takers[seeker_id][0]["lambda_taker"]
    for link_idx, link in takers[seeker_id].items():
        if link.setdefault("total_matching_rate", params["epsilon"]) == 0:
            continue
        seekers[seeker_id]["ride_distance_for_taker"] += link["lambda_taker"] * link["p_taker"] / lambda_become_taker * link.setdefault("total_ride_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])
        seekers[seeker_id]["detour_distance_for_taker"] += link["lambda_taker"] * link["p_taker"] / lambda_become_taker * link.setdefault("total_detour_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])
        seekers[seeker_id]["shared_distance_for_taker"] += link["lambda_taker"] * link["p_taker"] / lambda_become_taker * link.setdefault("total_shared_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])
        seekers[seeker_id]["saved_distance_for_taker"] += link["lambda_taker"] * link["p_taker"] / lambda_become_taker * link.setdefault("total_saved_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])

        seekers[seeker_id]["ride_distance"] += link["lambda_taker"] * link["p_taker"] / seekers[seeker_id]["lambda"] * link.setdefault("total_ride_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])
        seekers[seeker_id]["detour_distance"] += link["lambda_taker"] * link["p_taker"] / seekers[seeker_id]["lambda"] * link.setdefault("total_detour_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])
        seekers[seeker_id]["shared_distance"] += link["lambda_taker"] * link["p_taker"] / seekers[seeker_id]["lambda"] * link.setdefault("total_shared_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])
        seekers[seeker_id]["saved_distance"] += link["lambda_taker"] * link["p_taker"] / seekers[seeker_id]["lambda"] * link.setdefault("total_saved_distance", 0) / link.setdefault("total_matching_rate", params["epsilon"])
    
    seekers[seeker_id]["ride_distance"] += (1 - seekers[seeker_id]["matching_prob"]) * L
    seekers[seeker_id]["ride_distance_for_seeker"] += (1 - seekers[seeker_id]["p_seeker"]) * L
    seekers[seeker_id]["ride_distance_for_taker"] += takers[seeker_id][len(takers[seeker_id]) - 1]["lambda_taker"] * (1 - takers[seeker_id][len(takers[seeker_id]) - 1]["p_taker"]) / lambda_become_taker * L


# ---------- calculate the mean vacant vehicle pick-up time of each OD pair ----------
# the stopping rate of vehicles at nodes i
v_dic = {}
for node_id in node_dict.keys():
    W_node_id = []
    for OD_index, value in OD_dict.items():
        if value[1] == node_id:
            W_node_id.append(OD_index)
    M_node_id = match_df[match_df['destination'] == node_id]

    v_i = 0

    for OD_index in W_node_id:
        P_w = seekers[OD_index]['matching_prob']
        lambda_w = seekers[OD_index]['lambda']
        v_i += (1-P_w) * lambda_w

    for seeker_id, takers_of_seeker in matches.items():
        for taker in takers_of_seeker:
            if taker['destination'] == node_id:
                v_i += taker["eta_match"] * takers[taker["taker_id"]][taker["link_idx"]]["rho_taker"]

    v_dic[node_id] = v_i
# the number of vacant vehicles at each node i
n_i_v_dic = {}
for node_id in node_dict.keys():
    n_i_v = 0
    for node_id in node_dict.keys():
        n_i_v += v_dic[node_id]
    n_i_v_dic[node_id] = v_dic[node_id] / n_i_v * n_v

def pickup_distance_with_vacant_vehicles(n_vacant_vehicles):
    return n_vacant_vehicles / 100

# the total number of vacant vehicles at each instant
tmp = 0 
t_w_dic = []
for od_id in OD_dict.keys():
    L_w = seekers[od_id]["ride_distance"]
    E_w = seekers[od_id]["shared_distance"]
    lambda_w = seekers[od_id]['lambda']
    p_s_w = seekers[od_id]["p_seeker"]
    tmp += ( (L_w - E_w / 2 ) * lambda_w + t_w_pk_bar * (1 - p_s_w) * lambda_w)

    # the pick-up distance
    nearest_nodes = []
    for value in OD_dict.values():
        res = search_matching_OD(od_id, value)
        if res:
            nearest_nodes.append(res)
    n_vacant_vehicles = 0
    for node in nearest_nodes:
        n_vacant_vehicles += n_i_v_dic[node]

    t_w_pk_bar = pickup_distance_with_vacant_vehicles(n_vacant_vehicles)
    
    t_w_dic[od_id] = p_s_w * 0.5 * params['search_radius'] + (1 - p_s_w) * t_w_pk_bar

n_v = params['n_v'] - tmp

#  the mean ridepooling cost between each OD pair
def p_w_function(demand, solo_price):
    return demand * solo_price

C_w_dic = {}

for od_id in OD_dict.keys():
    C_w_dic[od_id] = params['beta'] * (t_w_dic[od_id] + seekers[od_id]["ride_distance"] / params['speed']) + \
    p_w_function(seekers[od_id]['lambda'], 1) + params['delta']

# ridepooling demand rate between OD pair
def f_w_function(C_w):
    return C_w 

lambda_w_dic = {}

for od_id in OD_dict.keys():
    lambda_w_dic[od_id] = f_w_function(seekers[od_id]['lambda'])


# ---------- Save the prediction result to csv ----------
print("Result saving ...")
result = pd.DataFrame.from_dict(seekers, orient='index').loc[:, [
    "matching_prob", "ride_distance", "detour_distance", "shared_distance", "saved_distance",
    "ride_distance_for_taker", "detour_distance_for_taker", "shared_distance_for_taker", "saved_distance_for_taker",
    "ride_distance_for_seeker", "detour_distance_for_seeker", "shared_distance_for_seeker", "saved_distance_for_seeker"]]
result.index.name = "OD_id"
result.to_csv("result/predict_result.csv")

# ---------- Dump to pickle ----------
f = open('variables/seekers.pickle', 'wb')
pickle.dump(seekers, f)
f.close()
f = open('variables/takers.pickle', 'wb')
pickle.dump(takers, f)
f.close()
f = open('variables/matches.pickle', 'wb')
pickle.dump(matches, f)
f.close()

# -------------------- End --------------------
end_time = time.time()
# ---------- Log ----------
with open("log.txt", "a") as f:
    f.write(time.ctime() + ": Run " + os.path.basename(__file__) + " with Params = " + str(params) + "; Cost " + str(end_time - start_time) + 's\n')
