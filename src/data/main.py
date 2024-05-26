import pickle
from copy import copy
from datasets import get_dataset_config_names

# def update_split_id():
#     with open('split_id_55.pckl', 'rb') as f:
#         split_id: dict = pickle.load(f)
#     with open('split_id_54.pckl', 'rb') as f:
#         split_id1 = pickle.load(f)
#         print(end='')
#     with open('split_id_56.pckl', 'rb') as f:
#         split_id2 = pickle.load(f)
#         print(end='')
#
#     id_path = []
#     id_path1 = []
#     id_path2 = []
#     tmp_split_id = dict()
#     tmp_split_id1 = dict()
#     tmp_split_id2 = dict()
#
#     result_d = dict()
#
#     for k,v in split_id2.items():
#         if len(v) == 50 and k != "batch_99999":
#             id_path2.extend(v)
#     new_metadata_volume = len(id_path2)
#     for idx_batch, i in enumerate(range(0, new_metadata_volume, 50),start=0):
#         tmp_split_id2[f"batch_%.5d" % (idx_batch,)] = id_path2[i:i + 50]
#     result_d.update(tmp_split_id2)
#
#     for k,v in split_id1.items():
#         if len(v) == 50 and k != "batch_99999":
#             id_path1.extend(v)
#     new_metadata_volume = len(id_path1)
#     for idx_batch, i in enumerate(range(0, new_metadata_volume, 50),start=len(tmp_split_id2)):
#         tmp_split_id1[f"batch_%.5d" % (idx_batch,)] = id_path1[i:i + 50]
#     result_d.update(tmp_split_id1)
#
#     for k,v in split_id.items():
#         if len(v) == 50 and k != "batch_99999":
#             id_path.extend(v)
#     new_metadata_volume = len(id_path)
#     for idx_batch, i in enumerate(range(0, new_metadata_volume, 50),start=len(tmp_split_id1)+len(tmp_split_id2)):
#         tmp_split_id[f"batch_%.5d" % (idx_batch,)] = id_path[i:i + 50]
#     result_d.update(tmp_split_id)
#     result_d['batch_99999'] = split_id['batch_99999']
#     tmp = []
#     for v in result_d.values():
#         tmp.extend(v)
#
#     assert 1150 + 6350 + 18700 + 2 == len(set(tmp))
#     with open('split_id.pckl', 'wb') as f:
#         pickle.dump(obj=result_d,file=f)

s = get_dataset_config_names('GSU24AI03-SU24AI21/downloaded-vietnamese-video',trust_remote_code=True)
print(sorted(s))