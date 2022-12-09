import json
import pickle
import numpy as np
from collections import OrderedDict

def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pickle_to_json(filename):
    ordered_dict = load_pickle(filename)
    json_obj = json.dumps(ordered_dict, cls=NumpyEncoder)
    with open(filename.replace('.pkl', '.json'), 'w') as f:
        f.write(json_obj)

def load_json(filename):
    with open(filename, 'r') as read_file:
        loaded_dict = json.loads(read_file.read())
    loaded_dict = OrderedDict(loaded_dict)
    for k, v in loaded_dict.items():
        loaded_dict[k] = np.asarray(v)
    return loaded_dict

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# save_pickle_to_json('data/layer_infos/convnext_layer_infos.pkl')
# save_pickle_to_json('data/layer_infos/resnet_layer_infos.pkl')
# save_pickle_to_json('data/layer_infos/mobilenet_layer_infos.pkl')

file = load_json('data/layer_infos/convnext_layer_infos.json')
print(type(file))
print(type(file['embeddings.patch_embeddings']))
