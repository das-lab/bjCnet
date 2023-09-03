import torch
import numpy as np
import dgl
import random
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs
from tqdm import tqdm
import os


class MyDataset4Train(DGLDataset):
    def __init__(self, processed_msg_path):
        super(MyDataset4Train, self).__init__("training_dataset")
        self.processed_msg_path = processed_msg_path

    def download(self):
        pass

    def process(self, graph_map_path):
        if self.has_cache():
            self.load()
            return

        self.dataset = []
        self.graph_bug = []
        self.graph_patch = []
        self.graph_clean = []
        with open(graph_map_path, "r", encoding="utf-8", errors="ignore") as r:
            graph_map_list = r.readlines()

        for graph_map in tqdm(graph_map_list):
            path_str = graph_map.split("|")
            if not os.path.exists(path_str[0].replace("\n", "")) or not os.path.exists(path_str[1].replace("\n", "")):
                continue

            g_b = self.__load_graph_info(path_str[0].replace("\n", ""))
            g_p = self.__load_graph_info(path_str[1].replace("\n", ""))
            g_c = self.__get_pureclean_graph(path_str[0].replace("\n", ""))

            if g_b is None or g_p is None or g_c is None:
                continue

            self.graph_bug.append(g_b)
            self.graph_patch.append(g_p)
            self.graph_clean.append(g_c)
            self.dataset.append((g_b, g_p, g_c))

        self.data_len = len(self.dataset)

    def __get_pureclean_graph(self, path):
        temp = path.split("\\")
        pureclean_dir = ""
        for idx in range(len(temp) - 3):
            pureclean_dir = os.path.join(pureclean_dir, temp[idx])

        pureclean_dir = os.path.join(pureclean_dir, "pure_clean")
        dir_list = []
        for root, dirs, files in os.walk(pureclean_dir):
            for file in files:
                dir_list.append(root)

        dir_list = list(set(dir_list))

        idx = random.randint(0, len(dir_list) - 1)

        return self.__load_graph_info(dir_list[idx])

    def __load_graph_info(self, path):
        AM_path = os.path.join(path, "AM.txt")
        node_features_path = os.path.join(path, "node_features_vector.txt")
        edge_features_path = os.path.join(path, "edge_features.txt")

        with open(AM_path, "r", encoding="utf-8", errors="ignore") as r:
            AM = r.readlines()

        with open(node_features_path, "r", encoding="utf-8", errors="ignore") as r:
            node_features = r.readlines()

        with open(edge_features_path, "r", encoding="utf-8", errors="ignore") as r:
            edge_features = r.readlines()

        U = []
        V = []
        for line in AM:
            _line = line.replace("\n", "").split(",")
            U.append(int(_line[0]))
            V.append(int(_line[1]))

        node_vector_list = []
        for line in node_features:
            line_int = []
            _line = line.split(",")
            for _l in _line:
                line_int.append(float(_l))
            node_vector_list.append(line_int)

        edge_vector_list = []
        for line in edge_features:
            edge_type = self.__convert_edge_feature(line.replace("\n", ""))
            edge_vector_list.append(edge_type)

        U = torch.tensor(U, dtype=torch.int32) - 1
        V = torch.tensor(V, dtype=torch.int32) - 1

        g = dgl.graph((U, V))
        g.ndata["X"] = torch.tensor(np.array(node_vector_list), dtype=torch.float32)
        g.edata["X"] = torch.tensor(np.array(edge_vector_list), dtype=torch.float32)

        return g

    def __convert_edge_feature(self, edge_feature):
        if edge_feature == "1,0,0,0,0,0,0":
            return 0
        if edge_feature == "0,1,0,0,0,0,0":
            return 1
        if edge_feature == "0,0,1,0,0,0,0":
            return 2
        if edge_feature == "0,0,0,1,0,0,0":
            return 3
        if edge_feature == "0,0,0,0,1,0,0":
            return 4
        if edge_feature == "0,0,0,0,0,1,0":
            return 5
        if edge_feature == "0,0,0,0,0,0,1":
            return 6

    def save(self):
        if self.has_cache():
            return

        graph_path = os.path.join(os.path.join(self.processed_msg_path, "bug_graphs_train.bin"))
        save_graphs(graph_path, self.graph_bug)

        graph_path = os.path.join(os.path.join(self.processed_msg_path, "patch_graphs_train.bin"))
        save_graphs(graph_path, self.graph_patch)

        graph_path = os.path.join(os.path.join(self.processed_msg_path, "clean_graphs_train.bin"))
        save_graphs(graph_path, self.graph_clean)

    def load(self):
        self.dataset = []
        bug_graphs, _ = load_graphs(os.path.join(self.processed_msg_path, 'bug_graphs_train.bin'))
        patch_graphs, _ = load_graphs(os.path.join(self.processed_msg_path, 'patch_graphs_train.bin'))
        clean_graphs, _ = load_graphs(os.path.join(self.processed_msg_path, 'clean_graphs_train.bin'))

        self.data_len = len(bug_graphs)
        for index in range(self.data_len):
            self.dataset.append((bug_graphs[index], patch_graphs[index], clean_graphs[index]))

    def has_cache(self):
        bug_data_path = os.path.join(self.processed_msg_path, "bug_graphs_train.bin")
        patch_data_path = os.path.join(self.processed_msg_path, "patch_graphs_train.bin")
        clean_data_path = os.path.join(self.processed_msg_path, "clean_graphs_train.bin")

        if os.path.exists(bug_data_path) and os.path.exists(patch_data_path) and os.path.exists(clean_data_path):
            return True

        return False

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.data_len


class MyDataset4Test(DGLDataset):
    def __init__(self, processed_msg_path):
        super(MyDataset4Test, self).__init__("testing_dataset")
        self.processed_msg_path = processed_msg_path

    def download(self):
        pass

    def process(self, graph_map_path, graph_path):
        if self.has_cache():
            self.load()
            return

        self.dataset = []
        self.label = []
        with open(graph_map_path, "r", encoding="utf-8", errors="ignore") as r:
            graph_map_list = r.readlines()

        for graph_map in tqdm(graph_map_list):
            path_str = graph_map.split("|")
            if not os.path.exists(path_str[0].replace("\n", "")) or not os.path.exists(path_str[1].replace("\n", "")):
                continue

            g_b = self.__load_graph_info(path_str[0].replace("\n", ""))
            g_c = self.__load_graph_info(path_str[1].replace("\n", ""))

            if g_b is None or g_c is None:
                continue

            self.dataset.append(g_b)
            self.label.append(1)
            self.dataset.append(g_c)
            self.label.append(0)

        pure_clean_path = []
        for root, dirs, files in os.walk(graph_path):
            for file in files:
                if root.find("pure_clean") > -1 and root.find("1.4") > -1:
                    pure_clean_path.append(root)

        pure_clean_path = list(set(pure_clean_path))

        for path in pure_clean_path:
            g_c = self.__load_graph_info(path)
            self.dataset.append(g_c)
            self.label.append(0)

        self.data_len = len(self.dataset)

    def __load_graph_info(self, path):
        AM_path = os.path.join(path, "AM.txt")
        node_features_path = os.path.join(path, "node_features_vector.txt")
        edge_features_path = os.path.join(path, "edge_features.txt")

        with open(AM_path, "r", encoding="utf-8", errors="ignore") as r:
            AM = r.readlines()

        with open(node_features_path, "r", encoding="utf-8", errors="ignore") as r:
            node_features = r.readlines()

        with open(edge_features_path, "r", encoding="utf-8", errors="ignore") as r:
            edge_features = r.readlines()

        U = []
        V = []
        for line in AM:
            _line = line.replace("\n", "").split(",")
            U.append(int(_line[0]))
            V.append(int(_line[1]))

        node_vector_list = []
        for line in node_features:
            line_int = []
            _line = line.split(",")
            for _l in _line:
                line_int.append(float(_l))
            node_vector_list.append(line_int)

        edge_vector_list = []
        for line in edge_features:
            edge_type = self.__convert_edge_feature(line.replace("\n", ""))
            edge_vector_list.append(edge_type)

        U = torch.tensor(U, dtype=torch.int32) - 1
        V = torch.tensor(V, dtype=torch.int32) - 1

        g = dgl.graph((U, V))
        g.ndata["X"] = torch.tensor(np.array(node_vector_list), dtype=torch.float32)
        g.edata["X"] = torch.tensor(np.array(edge_vector_list), dtype=torch.float32)

        return g

    def __convert_edge_feature(self, edge_feature):
        if edge_feature == "1,0,0,0,0,0,0":
            return 0
        if edge_feature == "0,1,0,0,0,0,0":
            return 1
        if edge_feature == "0,0,1,0,0,0,0":
            return 2
        if edge_feature == "0,0,0,1,0,0,0":
            return 3
        if edge_feature == "0,0,0,0,1,0,0":
            return 4
        if edge_feature == "0,0,0,0,0,1,0":
            return 5
        if edge_feature == "0,0,0,0,0,0,1":
            return 6

    def save(self):
        if self.has_cache():
            return

        graph_path = os.path.join(os.path.join(self.processed_msg_path, "test.bin"))
        self.label = torch.tensor(self.label, dtype=torch.float32)
        save_graphs(graph_path, self.dataset, {"labels": self.label})

    def load(self):
        self.dataset = []
        self.dataset, self.label = load_graphs(os.path.join(self.processed_msg_path, 'test.bin'))
        self.label = self.label["labels"]

        self.data_len = len(self.dataset)

    def has_cache(self):
        test_data_path = os.path.join(self.processed_msg_path, "test.bin")

        if os.path.exists(test_data_path):
            return True

        return False

    def __getitem__(self, index):
        return self.dataset[index], self.label[index]

    def __len__(self):
        return self.data_len
