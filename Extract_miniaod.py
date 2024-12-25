import uproot
import math
import numpy as np
import awkward as ak
from time import time
import pickle
import tqdm
from torch_geometric.data import Data
import torch

MISSING = -999999

########################################
# HGCAL Values                         #
########################################

HGCAL_X_Min = -36
HGCAL_X_Max = 36

HGCAL_Y_Min = -36
HGCAL_Y_Max = 36

HGCAL_Z_Min = 13
HGCAL_Z_Max = 265

HGCAL_Min = 0
HGCAL_Max = 230
# HGCAL_Max_EE =3200
# HGCAL_Max_FH = 2200
# HGCAL_MAX_AH =800


########################################
# ECAL Values                          #
########################################

X_Min = -150
X_Max = 150

Y_Min = -150
Y_Max = 150

Z_Min = -330
Z_Max = 330

Eta_Min = -3.0
Eta_Max = 3.0

Phi_Min = -np.pi
Phi_Max = np.pi

iEta_Min = -85
iEta_Max = 85

iPhi_Min = 1
iPhi_Max = 360

iX_Min = 1
iX_Max = 100

iY_Min = 1
iY_Max = 100

ECAL_Min = 0
ECAL_Max = 360


def setptetaphie(pt, eta, phi, e):
    pta = abs(pt)
    return np.array([np.cos(phi) * pta, np.sin(phi) * pt, np.sinh(eta) * pta, e])


def getMass(lvec):
    return np.sqrt(
        lvec[3] * lvec[3] - lvec[2] * lvec[2] - lvec[1] * lvec[1] - lvec[0] * lvec[0]
    )


def rescale(feature, minval, maxval):
    top = feature - minval
    bot = maxval - minval
    return top / bot


def dphi(phi1, phi2):
    dphi = np.abs(phi1 - phi2)
    gt = dphi > np.pi
    dphi[gt] = 2 * np.pi - dphi[gt]
    return dphi


def dR(eta1, eta2, phi1, phi2):
    dp = dphi(phi1, phi2)
    de = np.abs(eta1 - eta2)

    return np.sqrt(dp * dp + de * de)


def cartfeat(x, y, z, En, det=None):
    E = rescale(En, ECAL_Min, ECAL_Max)
    x = rescale(x, X_Min, X_Max)
    y = rescale(y, Y_Min, Y_Max)
    z = rescale(z, Z_Min, Z_Max)

    if det is None:
        return ak.concatenate(
            (
                x[:, :, None],
                y[:, :, None],
                z[:, :, None],
                E[:, :, None],
            ),
            -1,
        )
    else:
        return ak.concatenate(
            (
                x[:, :, None],
                y[:, :, None],
                z[:, :, None],
                E[:, :, None],
                det[:, :, None],
            ),
            -1,
        )


def torchify(feat, graph_x=None):
    data = [
        Data(x=torch.from_numpy(ak.to_numpy(ele).astype(np.float32))) for ele in feat
    ]
    if graph_x is not None:
        for d, gx in zip(data, graph_x):
            d.graph_x = gx
    return data


def npify(feat):
    t0 = time()
    data = [ak.to_numpy(ele) for ele in feat]
    print("took %f" % (time() - t0))
    return data


class Extract:
    def __init__(
        self,
        outfolder="pickles",
        path="merged.root",
        treeName="nTuplelize/T",
    ):
        if path is not None:
            # path = '~/shared/nTuples/%s'%path
            self.tree = uproot.open("%s:%s" % (path, treeName))

        self.outfolder = outfolder

    def read(self, kind, N=None):
        varnames = [
            "Ele_Gen_E",
            "Ele_Gen_Pt",
            "Ele_Gen_Eta",
            "Ele_Gen_Phi",
            "eta",
            "phi",
            "Hit_X_Ele1",
            "Hit_Y_Ele1",
            "Hit_Z_Ele1",
            "RecHitEnEle1",
            "Hit_X_Ele2",
            "Hit_Y_Ele2",
            "Hit_Z_Ele2",
            "RecHitEnEle2",
            "m_gen",
            "m_reco",
        ]

        print("Reading in %s..." % kind)
        arrs = self.tree.arrays(varnames)
        # Make sure that there are atleast 2 electrons at gen level
        # arrs = arrs[[len(j) >= 2 for j in arrs["Ele_Gen_Pt"]]]
        # arrs = arrs[[j[0] > 10 and j[1] > 10 for j in arrs["Ele_Gen_Pt"]]]
        # arrs = arrs[
        #    [abs(j[0]) < 1.4442 and abs(j[1]) < 1.4442 for j in arrs["Ele_Gen_Eta"]]
        # ]
        result = {}
        t0 = time()
        with open("%s/trueE_target.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(arrs["m_gen"], outpickle)
        with open("%s/trueE_reco.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(arrs["m_reco"], outpickle)
        print("\tDumping target took %f seconds" % (time() - t0))
        print("Building cartesian features..")
        HitsX = ak.concatenate((arrs["Hit_X_Ele1"], arrs["Hit_X_Ele2"]), axis=1)

        HitsY = ak.concatenate((arrs["Hit_Y_Ele1"], arrs["Hit_Y_Ele2"]), axis=1)
        HitsZ = ak.concatenate((arrs["Hit_Z_Ele1"], arrs["Hit_Z_Ele2"]), axis=1)
        HitsEn = ak.concatenate((arrs["RecHitEnEle1"], arrs["RecHitEnEle2"]), axis=1)
        ele1flg = arrs["Hit_X_Ele1"] * 0 + 1
        ele2flg = arrs["Hit_X_Ele2"] * 0
        eleflags = ak.concatenate((ele1flg, ele2flg), axis=1)
        print(ele1flg, ele2flg)

        cf = cartfeat(
            HitsX,
            HitsY,
            HitsZ,
            HitsEn,
        )
        print("\tBuilding features took %f seconds" % (time() - t0))
        t0 = time()
        result["cartfeat"] = torchify(cf)
        print("\tTorchifying took %f seconds" % (time() - t0))
        t0 = time()
        with open("%s/cartfeat.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["cartfeat"], f, pickle_protocol=4)
        with open("%s/eleflags.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(eleflags, outpickle)
        print("\tDumping took %f seconds" % (time() - t0))
        return result
