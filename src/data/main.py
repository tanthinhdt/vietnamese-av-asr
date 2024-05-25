import math
import pickle
from huggingface_hub import HfFileSystem
fs = HfFileSystem()


# with open('./split_id.pckl', 'rb') as f:
#     d: dict = pickle.load(f)
#
# d['batch_00181'].append(d['batch_99999'].pop(-1))
# d['batch_00181'].append(d['batch_99999'].pop(-1))
# d['batch_00181'].append(d['batch_99999'].pop(-1))
#
# with open('./split_id.pckl', 'wb') as f:
#     pickle.dump(d,f)
#
# with open('./split_id.pckl', 'rb') as f:
#     d: dict = pickle.load(f)
#
# print(d['batch_99999'])
