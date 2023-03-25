import os
import pandas as pd
import re
import difflib
from util.md5_helper import MD5_HELPER
from util.Config import Config
import gensim.downloader as api
import numpy as np
from tqdm import tqdm

raw_data_path = Config.raw_data_path
processed_data_path = Config.processed_data_path


class Process4PROMISE:
    # process PROMISE dataset
    def __init__(self):
        self.product = Config.product
        self.source_code_path = Config.source_code_path

        self.md5_helper = MD5_HELPER()

        self.report_files = []
        report_file_path = os.path.join(raw_data_path, "bug-data\\{}".format(self.product))
        for file in os.listdir(report_file_path):
            self.report_files.append(os.path.join(report_file_path, file))

        self.report_files.sort()

        self.version_list = []
        self.__get_version_list()

    def process(self):
        for index in range(len(self.report_files)):
            anchor = self.report_files[index]
            temp = anchor.split("\\")
            anchor_version = re.split("-|(\.csv)", temp[len(temp) - 1])[2]
            report_csv_df = pd.read_csv(anchor)
            key_info = report_csv_df[["name", "bug"]]

            for row in key_info.itertuples():
                file_name = getattr(row, "name")
                anchor_file_path = self.__get_source_code_file(file_name, anchor_version)

                if anchor_file_path is None:
                    print("can not get {}-{} file path".format(file_name, anchor_version))
                    continue

                save_path = os.path.join(processed_data_path, Config.product)
                save_path = os.path.join(save_path, anchor_version)

                bug = getattr(row, "bug")
                if bug > 0:
                    label = 1

                    result = self.__find_patch_file(anchor_file_path)
                else:
                    label = 0
                    with open(anchor_file_path, "r", errors="ignore") as r:
                        anchor_source_code = r.read()

                    result = anchor_source_code

                if result is None:
                    print("can not get {} file and its patch".format(anchor_file_path))
                    continue

                self.__save_source_code(save_path, label, file_name, anchor_version, result)

    def __save_source_code(self, path, label, file_name, anchor_version, save_content):
        anchor_source_code = None
        patch_source_code = None
        anchor_file_path = None
        patch_file_path = None

        file_name_md5 = self.md5_helper.get_md5(file_name)

        if label == 1:
            save_path = os.path.join(path, "bug")
            save_path = os.path.join(save_path, file_name_md5)

            anchor_file_path = os.path.join(save_path, "bug")
            patch_file_path = os.path.join(save_path, "patch")
            if not os.path.isdir(anchor_file_path):
                os.makedirs(anchor_file_path)
            if not os.path.isdir(patch_file_path):
                os.makedirs(patch_file_path)

            a_file_name = "bug.java"
            p_file_name = "patch.java"

            anchor_source_code = save_content["anchor_source_code"]
            patch_source_code = save_content["candidate_source_code"]
        else:
            save_path = os.path.join(path, "pure_clean")
            save_path = os.path.join(save_path,file_name_md5)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            a_file_name = "pureclean.java"

        if label == 1:
            with open(os.path.join(anchor_file_path, a_file_name), "w", encoding="utf-8", errors="ignore") as w:
                w.write(anchor_source_code)

            with open(os.path.join(patch_file_path, p_file_name), "w", encoding="utf-8", errors="ignore") as w:
                w.write(patch_source_code)
        else:
            with open(os.path.join(save_path, a_file_name), "w", encoding="utf-8", errors="ignore") as w:
                w.write(save_content)

    def __remove_annotation(self, source_code):
        pattern_str_annotate1 = "/\*{1,2}[\s\S]*?\*/"
        pattern_str_annotate2 = "//[\s\S]*?\n"

        source_code = re.sub(pattern_str_annotate1, "", source_code)
        source_code = re.sub(pattern_str_annotate2, "", source_code)

        return source_code

    def __find_patch_file(self, anchor_file_path):
        candidate_files = []
        temp = anchor_file_path.split("\\")[7]
        temp1 = temp.split("-")

        version_str = ""
        for i in range(len(temp1) - 1):
            version_str = version_str + "-" + temp1[i]
        version_str = version_str[1:] + "-"

        for version in self.version_list:
            if temp.find(version) > -1 or version < temp1[len(temp1) - 1]:
                continue

            temp2 = anchor_file_path
            candidate_files.append(temp2.replace(temp, version_str + version))

        with open(anchor_file_path, "r", errors="ignore") as r:
            anchor_source_code = r.read()

        for file in candidate_files:
            if not os.path.exists(file):
                continue

            with open(file, "r", errors="ignore") as r1:
                candidate_source_code = r1.read()

            if self.__is_different(anchor_source_code, candidate_source_code):
                return {
                    "anchor_source_code": anchor_source_code,
                    "candidate_source_code": candidate_source_code
                }

        return None

    def __is_different(self, anchor_source_code, candidate_source_code):
        differ = difflib.unified_diff(anchor_source_code, candidate_source_code, fromfile="anchor", tofile="candidate")

        changes = ""
        while True:
            try:
                changes = changes + next(differ)
            except:
                break

        if len(changes) == 0:
            return False

        return True

    def __get_version_list(self):
        for i in self.report_files:
            temp = i.split("\\")
            version = re.split("-|(\.csv)", temp[len(temp) - 1])[2]
            self.version_list.append(version)

    def __get_source_code_file(self, name, anchor_version):
        temp = name.split(".")
        abs_file_path = ""
        for t in temp:
            abs_file_path = abs_file_path + "\\" + t

        abs_file_path = abs_file_path[1:]

        for root, dirs, files in os.walk(self.source_code_path):
            for file in files:
                temp1 = os.path.join(root, file)
                if root.find(anchor_version) > -1 and file.find(".java") > -1 and temp1.find(abs_file_path) > -1:
                    target_file_path = temp1
                    return target_file_path

        return None


class Process4MSG:
    def __init__(self):
        self.node_feature_list = []

        self.graph_path = Config.graph_path
        for root, dirs, files in os.walk(self.graph_path):
            for file in files:
                if file == "node_features_raw.txt":
                    self.node_feature_list.append(os.path.join(root, file))

        print("load pre-trained word vector")
        self.word2vec_model = api.load("glove-wiki-gigaword-300")
        self.word_to_idx()
        self.idx_to_word()
        self.get_embedding_matrix()
        self.embedding_len = self.word2vec_model.vector_size
        print("\nload completed")

        self.split_str = ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "+", "=", "`", "~", "\"", "'",
                          "|", "\\", "/", "?", ",", ".", "<", ">", " "]

    def word_to_idx(self):
        self.word2idx = self.word2vec_model.key_to_index

    def idx_to_word(self):
        self.idx2word = self.word2vec_model.index_to_key

    def get_embedding_matrix(self):
        self.embedding_matrix = self.word2vec_model.vectors

        UNK_vector = np.zeros((1, 300)) - 1
        self.embedding_matrix = np.concatenate((self.embedding_matrix, UNK_vector))
        self.word2idx["<UNK>"] = len(self.word2idx)
        self.idx2word.append("<UNK>")

        PAD_vector = np.zeros((1, 300))
        self.embedding_matrix = np.concatenate((self.embedding_matrix, PAD_vector))
        self.word2idx["<PAD>"] = len(self.word2idx)
        self.idx2word.append("<PAD>")

    def get_wordvec(self, word):
        if len(word) == 0:
            return None

        if not (word in self.word2idx.keys()):
            word = "<UNK>"

        wordidx = self.word2idx[word]
        word_vector = self.embedding_matrix[wordidx]
        return word_vector

    def process(self):
        with tqdm(total=len(self.node_feature_list), desc='(T)') as pbar:
            for file in self.node_feature_list:
                with open(file, "r", encoding="utf-8", errors="ignore") as r:
                    node_features = r.readlines()

                tokens_list = []
                for node_feature in node_features:
                    token_list = self.__get_tokens(node_feature.replace("\n", ""))
                    tokens_list.append(token_list)

                save_path = file.replace("node_features_raw", "node_features_vector")
                self.__get_line_vector_by_sum(tokens_list, save_path)

                pbar.update()

    def __save_node_feature_vector(self, line_vector_list, save_path):
        with open(save_path, "a", encoding="utf-8", errors="ignore") as w:
            content = ""
            for line_vector in line_vector_list:
                _line_vector = ""
                for value in line_vector:
                    _line_vector = _line_vector + "," + str(value)

                _line_vector = _line_vector[1:] + "\n"
                content = content + _line_vector

            content = content[:len(content) - 1]
            w.write(content)

    def __get_line_vector_by_sum(self, tokens_list, save_path):
        line_vector_list = []
        for token_list in tokens_list:
            line_vector = np.zeros(300)
            for token in token_list:
                word_vector = self.get_wordvec(token)
                line_vector = line_vector + word_vector

            line_vector_list.append(line_vector.tolist())

        self.__save_node_feature_vector(line_vector_list, save_path)

    def __get_tokens(self, node_feature):
        token_list = []
        token = ""
        for i in range(len(node_feature)):
            str = node_feature[i]

            if str in self.split_str:
                if token != "":
                    new_token_list = self.__dismantle_camel_case(token)

                    for new_token in new_token_list:
                        token_list.append(new_token)

                    token = ""

                if str != " ":
                    token_list.append(str)
            else:
                token = token + str

        if token != "":
            new_token_list = self.__dismantle_camel_case(token)
            for new_token in new_token_list:
                token_list.append(new_token)

        return token_list

    def __dismantle_camel_case(self, raw_token):
        new_token_list = []
        token = ""

        i = 0
        while i < len(raw_token):
            if raw_token[i] >= "A" and raw_token[i] <= "Z":
                if token != "":
                    if i > 0 and raw_token[i - 1] >= "A" and raw_token[i - 1] <= "Z":
                        token = token + raw_token[i]
                    else:
                        new_token_list.append(token.lower())
                        token = raw_token[i]
                else:
                    token = raw_token[i]
            else:
                token = token + raw_token[i]

            i = i + 1

        new_token_list.append(token.lower())
        return new_token_list


def main():
    # process = Process4PROMISE()
    process = Process4MSG()
    process.process()


if __name__ == "__main__":
    main()
