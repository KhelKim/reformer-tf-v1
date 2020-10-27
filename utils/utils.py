import json
import numpy as np
import random
import tensorflow as tf
from bunch import Bunch


def read_txt(path):
    with open(path) as f:
        result = f.readlines()
    return result


def read_json(path):
    with open(path) as j:
        result = json.load(j)
    return result


def read_npy(path):
    return np.load(path)


def save_json(path, obj):
    with open(path, 'w') as j:
        json.dump(obj, j, ensure_ascii=False)


def save_npy(path, obj):
    path = path.replace(".npy", "")
    np.save(path, obj)


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)


def str2bool(string):
    string = string.lower()
    return True if string in ["y", "t", "true", "yes"] else False


def make_dot_dic_from_dic(dic):
    doc_dic = Bunch()
    for key, value in dic.items():
        if isinstance(value, dict):
            doc_dic[key] = make_dot_dic_from_dic(value)
        else:
            doc_dic[key] = value
    return doc_dic


def make_dic_from_dot_dic(dot_dic):
    dic = {}
    for key in dot_dic.keys():
        value = dot_dic[key]
        if isinstance(value, Bunch):
            dic[key] = make_dic_from_dot_dic(value)
        else:
            dic[key] = value
    return dic


if __name__ == "__main__":
    set_random_seed(3)
