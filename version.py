import numpy as np
import pickle

# Assuming data_dict is your dictionary containing 'data' and 'labels'
data_dict = pickle.load(open('./data.pickle', 'rb')) 

data = data_dict['data']
labels = data_dict['labels']

for i, item in enumerate(data):
    if isinstance(item, np.ndarray):
        print(f"Shape of data item {i}: {item.shape}")
    elif isinstance(item, list):
        print(f"Length of data item {i}: {len(item)}")
    else:
        print(f"Unknown type for data item {i}")

# Optionally, you can also check the shape of labels
print(f"Shape of labels: {labels.shape}")