import os
import shutil
# files = os.listdir('data/ODs/')

# for file in files:
#     print(file)
#     if file == 'OD_update location1.csv':
#         continue
#     shutil.copyfile('data/ODs/' + file,'data/OD.csv')

# cmd = 'python.exe .\generate_pickle.py'
# os.system(cmd)

# cmd = 'python.exe .\parallel_shortest_path_and_ego_graph.py'
# os.system(cmd)

# cmd = 'python.exe .\parallel_searching_of_matching_pairs.py'
# os.system(cmd)

# cmd = 'python.exe .\predict.py'
# os.system(cmd)

#     os.rename('result/predict_result.csv', 'result/predict_result_'+file)
cmd = 'python.exe .\generate_pickle.py'
os.system(cmd)

cmd = 'python.exe .\parallel_shortest_path_and_ego_graph.py'
os.system(cmd)

cmd = 'python.exe .\parallel_searching_of_matching_pairs.py'
os.system(cmd)

cmd = 'python.exe .\predict.py'
os.system(cmd)
