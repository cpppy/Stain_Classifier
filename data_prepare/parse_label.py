import os
import numpy as np

import config
import csv


def read_csv_file(label_fpath):
    with open(label_fpath, encoding='utf-8', newline='') as f:
        csv_reader = csv.reader(f)
        lines = [line for line in csv_reader]
        title_line = lines[0]
        data_lines = lines[1:]
        # print(title_line)
        print('isic_2019_sample_num: ', len(data_lines))  # 25331
        return title_line, data_lines


def get_isic_2019_labels(n_choose=10):
    label_fpath = config.label_fpath
    title_line, data_lines = read_csv_file(label_fpath)
    label_dict = {}
    cls_name_list = title_line[1:]
    print('cls_names: ', cls_name_list)
    data_lines = data_lines[0:n_choose]
    for line in data_lines:
        img_head = line[0]
        proba_seq = [float(proba) for proba in line[1:]]
        cls_name = cls_name_list[np.argmax(proba_seq)]
        # print(img_head, np.argmax(proba_seq), cls_name)
        label_dict[img_head] = cls_name
    return label_dict




if __name__=='__main__':

    get_isic_2019_labels()





