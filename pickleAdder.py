import torch
from torch_geometric.data import Data
import pickle
import numpy as np
import awkward as ak

targetE = []
targetEBM = []
targetAngle = []
inputs = []
for i in range(100, 1000, 100):
    inputs += torch.load("pickles_AToGG_M" + str(i) + "/cartfeat.pickle")
    targetE += list(
        np.load("pickles_AToGG_M" + str(i) + "/trueE_target.pickle", allow_pickle=True)
    )
    targetEBM += list(
        np.load(
            "pickles_AToGG_M" + str(i) + "/trueEBMScaled_target.pickle",
            allow_pickle=True,
        )
    )
    targetAngle += list(
        np.load(
            "pickles_AToGG_M" + str(i) + "/trueAngleScaled_target.pickle",
            allow_pickle=True,
        )
    )


def torchify(feat, graph_x=None):
    data = [Data(x=torch.from_numpy(np.array(Pho).astype(np.float32))) for Pho in feat]
    if graph_x is not None:
        for d, gx in zip(data, graph_x):
            d.graph_x = gx
    return data


split = 0.8

# cartfeat = torchify(inputs)
with open("pickles_combined/cartfeat.pickle", "wb") as f:
    torch.save(inputs, f, pickle_protocol=4)
with open("pickles_combined/trueE_target.pickle", "wb") as outpickle:
    pickle.dump(ak.Array(targetE), outpickle)
with open("pickles_combined/trueEBMScaled_target.pickle", "wb") as outpickle:
    pickle.dump(ak.Array(targetEBM), outpickle)
with open("pickles_combined/trueAngleScaled_target.pickle", "wb") as outpickle:
    pickle.dump(ak.Array(targetAngle), outpickle)
length = len(targetE)

# create train/test split
train_idx = np.random.choice(length, int(split * length + 0.5), replace=False)

mask = np.ones(length, dtype=bool)
mask[train_idx] = False
valid_idx = mask.nonzero()[0]

with open("pickles_combined/all_valididx.pickle", "wb") as f:
    pickle.dump(valid_idx, f)

with open("pickles_combined/all_trainidx.pickle", "wb") as f:
    pickle.dump(train_idx, f)
