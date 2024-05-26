import pickle
from copy import copy
from datasets import get_dataset_config_names

with open('./databases/split_id.pckl','rb') as f:
    split_id: dict = pickle.load(f)

tmp = []
for v in split_id.values():
    tmp.extend(v)
print(len(tmp))