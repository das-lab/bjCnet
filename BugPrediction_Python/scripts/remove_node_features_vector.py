import os
from util.Config import Config
from tqdm import tqdm

remove_list = []
for root, dirs, files in os.walk(Config.graph_path):
    for file in files:
        if file.find("node_features_vector")>-1:
            remove_list.append(os.path.join(root,file))

with tqdm(total=len(remove_list), desc='(T)') as pbar:
    for file in remove_list:
        os.remove(file)
        pbar.update()