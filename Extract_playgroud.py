import uproot
import math
from numba import jit
import numpy as np
import awkward as ak
from time import time
import pickle
import tqdm
from torch_geometric.data import Data
import torch
from ROOT import TLorentzVector

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
ECAL_Max = 250


def setptetaphie(pt, eta, phi, e):
    pta = abs(pt)
    return np.array([math.cos(phi) * pta, math.sin(phi) * pt, math.sinh(eta) * pta, e])


def getMass(lvec):
    return lvec[3] * lvec[3] - lvec[2] * lvec[2] - lvec[1] * lvec[1] - lvec[0] * lvec[0]


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


def cartfeat_HGCAL(z, En):
    # frac =  ((z<54)*0.0105) + (np.logical_and(z>54, z<154)*0.0789) + ((z>154)*0.0316)
    # ((z<54)*0.035) + ((z>54)*0.095) #(np.logical_and(z>54, z<154)*0.0789) + ((z>154)*0.0316)
    #    En = En*frac
    # if(z<54):
    #     HGCAL_Max=3200
    # elif(z>54 and z<154):
    #     HGCAL_Max=2200
    # elif(z>154):
    #     HGCAL_Max=850
    # HGCAL_Max = (z<54)*3300 +  (np.logical_and(z>54, z<154)*2500) + ((z>154)*900)
    E = rescale(En, HGCAL_Min, HGCAL_Max)
    # x = rescale(x, HGCAL_X_Min, HGCAL_X_Max)
    # y = rescale(y, HGCAL_Y_Min, HGCAL_Y_Max)
    z = rescale(z, HGCAL_Z_Min, HGCAL_Z_Max)

    # return ak.concatenate((x[:,:,None], y[:,:,None], z[:,:,None], E[:,:,None]), -1)
    return ak.concatenate((z[:, :, None], E[:, :, None]), -1)


def cartfeat(x, y, z, En, frac, det=None):
    E = rescale(En * frac, ECAL_Min, ECAL_Max)
    x = rescale(x, X_Min, X_Max)
    y = rescale(y, Y_Min, Y_Max)
    z = rescale(z, Z_Min, Z_Max)

    if det is None:
        return ak.concatenate(
            (x[:, :, None], y[:, :, None], z[:, :, None], E[:, :, None]), -1
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


def projfeat(eta, phi, z, En, frac, det=None):
    E = rescale(En * frac, ECAL_Min, ECAL_Max)
    eta = rescale(eta, Eta_Min, Eta_Max)
    phi = rescale(phi, Phi_Min, Phi_Max)
    z = rescale(z, Z_Min, Z_Max)

    if det is None:
        return ak.concatenate(
            (eta[:, :, None], phi[:, :, None], z[:, :, None], E[:, :, None]), -1
        )
    else:
        return ak.concatenate(
            (
                eta[:, :, None],
                phi[:, :, None],
                z[:, :, None],
                E[:, :, None],
                det[:, :, None],
            ),
            -1,
        )


def localfeat(i1, i2, z, En, frac, det=None):
    """
    In the barrel:
        i1 = iEta
        i2 = iPhi
    In the endcaps:
        i1 = iX
        i2 = iY
    """

    if det is not None:
        print("Error: local coordinates not defined for ES")
        return

    E = rescale(En * frac, ECAL_Min, ECAL_Max)

    Zfirst = ak.firsts(z)
    barrel = np.abs(Zfirst) < 300  # this is 1 if we are in the barrel, 0 in the endcaps

    xmax = barrel * iEta_Max + ~barrel * iX_Max
    xmin = barrel * iEta_Min + ~barrel * iX_Min

    ymax = barrel * iPhi_Max + ~barrel * iY_Max
    ymin = barrel * iPhi_Min + ~barrel * iY_Min

    x = rescale(i1, xmin, xmax)
    y = rescale(i2, ymin, ymax)

    whichEE = 2 * (Zfirst > 300) - 1  # +1 to the right of 0, -1 to the left of 0

    iZ = whichEE * ~barrel  # 0 in the barrel, -1 in left EE, +1 in right EE

    iZ, _ = ak.broadcast_arrays(iZ, x)

    return ak.concatenate(
        (x[:, :, None], y[:, :, None], iZ[:, :, None], E[:, :, None]), -1
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


varlists = {
    "BDTvars": [
        "Pho_R9",  #'Pho_S4', S4 is not populated
        "Pho_SigIEIE",
        "Pho_SigIPhiIPhi",
        "Pho_SCEtaW",
        "Pho_SCPhiW",
        #'Pho_CovIEtaIEta', 'Pho_CovIEtaIPhi','Pho_ESSigRR', not populated
        "Pho_SCRawE",
        "Pho_SC_ESEnByRawE",
        "Pho_HadOverEm",
        "eta",
        "phi",
        "Pho_Gen_Eta",
        "Pho_Gen_Phi",
        "iEtaPho1",
        "iEtaPho2",
        "Hit_Z_Pho1",
        "Hit_Z_Pho2",
        "Pho_Gen_E",
    ],
    "gun_pho": [
        "nPhotons",
        "Pho_Gen_E",
        "Pho_Gen_Eta",
        "Pho_Gen_Phi",
        "Pho_SCRawE",
        "pt",
        "eta",
        "phi",
        "Pho_R9",
        "Pho_HadOverEm",
        "rho",
        "iEtaPho1",
        "iEtaPho2",
        "iPhiPho1",
        "iPhiPho2",
        "Hit_ES_Eta_Pho1",
        "Hit_ES_Eta_Pho2",
        "Hit_ES_Phi_Pho1",
        "Hit_ES_Phi_Pho2",
        "Hit_ES_X_Pho1",
        "Hit_ES_X_Pho2",
        "Hit_ES_Y_Pho1",
        "Hit_ES_Y_Pho2",
        "Hit_ES_Z_Pho1",
        "Hit_ES_Z_Pho2",
        "ES_RecHitEnPho1",
        "ES_RecHitEnPho2",
        "Hit_Eta_Pho1",
        "Hit_Eta_Pho2",
        "Hit_Phi_Pho1",
        "Hit_Phi_Pho2",
        "Hit_X_Pho1",
        "Hit_X_Pho2",
        "Hit_Y_Pho1",
        "Hit_Y_Pho2",
        "Hit_Z_Pho1",
        "Hit_Z_Pho2",
        "RecHitEnPho1",
        "RecHitEnPho2",
        "RecHitFracPho1",
        "RecHitFracPho2",
        #'passLooseId', 'passMediumId', 'passTightId',
        "energy",
    ],
    "Hgg": [
        "nPhotons",
        "Pho_SCRawE",
        "eta",
        "phi",
        "Pho_R9",
        "Pho_HadOverEm",
        "rho",
        "iEtaPho1",
        "iEtaPho2",
        "iPhiPho1",
        "iPhiPho2",
        "Hit_ES_Eta_Pho1",
        "Hit_ES_Eta_Pho2",
        "Hit_ES_Phi_Pho1",
        "Hit_ES_Phi_Pho2",
        "Hit_ES_X_Pho1",
        "Hit_ES_X_Pho2",
        "Hit_ES_Y_Pho1",
        "Hit_ES_Y_Pho2",
        "Hit_ES_Z_Pho1",
        "Hit_ES_Z_Pho2",
        "ES_RecHitEnPho1",
        "ES_RecHitEnPho2",
        "Hit_Eta_Pho1",
        "Hit_Eta_Pho2",
        "Hit_Phi_Pho1",
        "Hit_Phi_Pho2",
        "Hit_X_Pho1",
        "Hit_X_Pho2",
        "Hit_Y_Pho1",
        "Hit_Y_Pho2",
        "Hit_Z_Pho1",
        "Hit_Z_Pho2",
        "RecHitEnPho1",
        "RecHitEnPho2",
        "RecHitFracPho1",
        "RecHitFracPho2",
        #'passLooseId', 'passMediumId', 'passTightId',
        "energy",
    ],
    "gun_30M": [
        "nElectrons",
        "Ele_Gen_E",
        "Ele_Gen_Eta",
        "Ele_Gen_Phi",
        "Ele_SCRawE",
        "eta",
        "phi",
        "Ele_R9",
        "Ele_HadOverEm",
        "rho",
        "iEtaEle1",
        "iEtaEle2",
        "iPhiEle1",
        "iPhiEle2",
        "Hit_ES_Eta_Ele1",
        "Hit_ES_Eta_Ele2",
        "Hit_ES_Phi_Ele1",
        "Hit_ES_Phi_Ele2",
        "Hit_ES_X_Ele1",
        "Hit_ES_X_Ele2",
        "Hit_ES_Y_Ele1",
        "Hit_ES_Y_Ele2",
        "Hit_ES_Z_Ele1",
        "Hit_ES_Z_Ele2",
        "ES_RecHitEnEle1",
        "ES_RecHitEnEle2",
        "Hit_Eta_Ele1",
        "Hit_Eta_Ele2",
        "Hit_Phi_Ele1",
        "Hit_Phi_Ele2",
        "Hit_X_Ele1",
        "Hit_X_Ele2",
        "Hit_Y_Ele1",
        "Hit_Y_Ele2",
        "Hit_Z_Ele1",
        "Hit_Z_Ele2",
        "RecHitEnEle1",
        "RecHitEnEle2",
        "RecHitFracEle1",
        "RecHitFracEle2",
        "passLooseId",
        "passMediumId",
        "passTightId",
        "energy_ecal_mustache",
    ],
    "gun_v3": [
        "nElectrons",
        "Ele_Gen_E",
        "Ele_Gen_Eta",
        "Ele_Gen_Phi",
        "Ele_SCRawE",
        "eta",
        "phi",
        "Ele_R9",
        "Ele_HadOverEm",
        "rho",
        "iEtaEle1",
        "iEtaEle2",
        "iPhiEle1",
        "iPhiEle2",
        "Hit_ES_Eta_Ele1",
        "Hit_ES_Eta_Ele2",
        "Hit_ES_Phi_Ele1",
        "Hit_ES_Phi_Ele2",
        "Hit_ES_X_Ele1",
        "Hit_ES_X_Ele2",
        "Hit_ES_Y_Ele1",
        "Hit_ES_Y_Ele2",
        "Hit_ES_Z_Ele1",
        "Hit_ES_Z_Ele2",
        "ES_RecHitEnEle1",
        "ES_RecHitEnEle2",
        "Hit_Eta_Ele1",
        "Hit_Eta_Ele2",
        "Hit_Phi_Ele1",
        "Hit_Phi_Ele2",
        "Hit_X_Ele1",
        "Hit_X_Ele2",
        "Hit_Y_Ele1",
        "Hit_Y_Ele2",
        "Hit_Z_Ele1",
        "Hit_Z_Ele2",
        "RecHitEnEle1",
        "RecHitEnEle2",
        "RecHitFracEle1",
        "RecHitFracEle2",
    ],
    "Zee_data": [
        "nElectrons",
        "Ele_SCRawE",
        "eta",
        "phi",
        "Ele_R9",
        "Ele_HadOverEm",
        "rho",
        "iEtaEle1",
        "iEtaEle2",
        "iPhiEle1",
        "iPhiEle2",
        "Hit_ES_Eta_Ele1",
        "Hit_ES_Eta_Ele2",
        "Hit_ES_Phi_Ele1",
        "Hit_ES_Phi_Ele2",
        "Hit_ES_X_Ele1",
        "Hit_ES_X_Ele2",
        "Hit_ES_Y_Ele1",
        "Hit_ES_Y_Ele2",
        "Hit_ES_Z_Ele1",
        "Hit_ES_Z_Ele2",
        "ES_RecHitEnEle1",
        "ES_RecHitEnEle2",
        "Hit_Eta_Ele1",
        "Hit_Eta_Ele2",
        "Hit_Phi_Ele1",
        "Hit_Phi_Ele2",
        "Hit_X_Ele1",
        "Hit_X_Ele2",
        "Hit_Y_Ele1",
        "Hit_Y_Ele2",
        "Hit_Z_Ele1",
        "Hit_Z_Ele2",
        "RecHitEnEle1",
        "RecHitEnEle2",
        "RecHitFracEle1",
        "RecHitFracEle2",
        "energy_ecal_mustache",
        "Ele_Gen_E",
        "Ele_Gen_Eta",
        "Ele_Gen_Phi",
        "Ele_Gen_Pt",
    ],
    "Zee_MC": [
        "nElectrons",
        "Ele_Gen_E",
        "Ele_Gen_Eta",
        "Ele_Gen_Phi",
        "Ele_SCRawE",
        "eta",
        "phi",
        "Ele_R9",
        "Ele_HadOverEm",
        "rho",
        "iEtaEle1",
        "iEtaEle2",
        "iPhiEle1",
        "iPhiEle2",
        "Hit_ES_Eta_Ele1",
        "Hit_ES_Eta_Ele2",
        "Hit_ES_Phi_Ele1",
        "Hit_ES_Phi_Ele2",
        "Hit_ES_X_Ele1",
        "Hit_ES_X_Ele2",
        "Hit_ES_Y_Ele1",
        "Hit_ES_Y_Ele2",
        "Hit_ES_Z_Ele1",
        "Hit_ES_Z_Ele2",
        "ES_RecHitEnEle1",
        "ES_RecHitEnEle2",
        "Hit_Eta_Ele1",
        "Hit_Eta_Ele2",
        "Hit_Phi_Ele1",
        "Hit_Phi_Ele2",
        "Hit_X_Ele1",
        "Hit_X_Ele2",
        "Hit_Y_Ele1",
        "Hit_Y_Ele2",
        "Hit_Z_Ele1",
        "Hit_Z_Ele2",
        "RecHitEnEle1",
        "RecHitEnEle2",
        "RecHitFracEle1",
        "RecHitFracEle2",
        "energy_ecal_mustache",
    ],
}

gun_readcut = "nElectrons>0"
gun_pho_readcut = "nPhotons>0"
Zee_readcut = "nElectrons==2"

readcuts = {
    "gun_30M": gun_readcut,
    "gun_v3": gun_readcut,
    "Zee_data": Zee_readcut,
    "Zee_MC": gun_readcut,
    "gun_pho": gun_pho_readcut,
    "BDTvars": gun_pho_readcut,
    "Hgg": "nPhotons==2",
}


def gun_savecut(result):
    return np.logical_and(result["Ele_Gen_E"] < 300, result["Ele_Gen_E"] > 5)


def gun_pho_savecut(result):
    return np.logical_and(result["Pho_Gen_E"] < 300, result["Pho_Gen_E"] > 5)


def Zee_savecut(result):
    return np.ones(result["phi"].shape, dtype=bool)


savecuts = {
    "gun_30M": gun_savecut,
    "gun_v3": gun_savecut,
    "Zee_data": Zee_savecut,
    "Zee_MC": Zee_savecut,
    "gun_pho": gun_pho_savecut,
    "BDTvars": gun_pho_savecut,
    "Hgg": Zee_savecut,
}

hasgen = {
    "gun_30M": True,
    "gun_v3": True,
    "Zee_data": False,
    "Zee_MC": True,
    "gun_pho": True,
    "BDTvars": True,
    "Hgg": False,
}

isEle = {
    "gun_30M": True,
    "gun_v3": True,
    "Zee_data": True,
    "Zee_MC": True,
    "gun_pho": False,
    "BDTvars": False,
    "Hgg": False,
}

nTuple = "./ElectronRecHits_ntuple_119.root"
# name of nTuple tree
treeName = "nTuplelize/T;5"
# path to folder in which to store extracted python-ready data objects. Should be somewhere in shared/pickles
outfolder = "pickles"
# nTuple-handling is wrapped in Extract.py
kind = "Zee_data"
varnames = varlists[kind]
readcut = readcuts[kind]
tree = uproot.open("%s:%s" % (nTuple, treeName))
t0 = time()
print("Reading in %s..." % kind)
N = None
arrs = tree.arrays(varnames, readcut, entry_stop=N)
arrs = arrs[[len(j) >= 2 for j in arrs["Ele_Gen_Pt"]]]
print(arrs)
gen = []
reco = []
event = []
hits = []

result = {}

for var in arrs.fields:
    if var[-4:-1] == "Ele" or var[-4:-1] == "Pho":  # hit level information
        # different from reco: branches named "ele1" "ele2"
        name = var[:-4]
        hits.append(name)
        continue
    elif var[:7] == "Ele_Gen" or var[:7] == "Pho_Gen":  # gen level information
        gen.append(var)
    elif (
        var == "rho" or var == "nElectrons" or var == "nPhotons"
    ):  # event level information
        event.append(var)
    else:  # reco level information
        reco.append(var)

print("\tio took %f seconds" % (time() - t0))


def gen_match(phigen, phireco, etagen, etareco, threshold=0.05):
    idxs = ak.argcartesian((etagen, etareco), axis=1)

    # awkward insists that I index the cartesitna product pairs with '0' and '1' rather than ints
    genetas = etagen[idxs[:, :, "0"]]
    recoetas = etareco[idxs[:, :, "1"]]

    genphis = phigen[idxs[:, :, "0"]]
    recophis = phireco[idxs[:, :, "1"]]

    dphis = np.abs(genphis - recophis)
    gt = dphis > np.pi
    # you can't assign to awkward arrays in place, so this is an inefficient hack
    dphis = gt * (2 * np.pi - dphis) + (1 - gt) * (dphis)

    detas = np.abs(genetas - recoetas)

    dR2s = dphis * dphis + detas * detas

    matched = dR2s < threshold * threshold

    return idxs[matched]  # these are (gen, reco) index pairs


if hasgen[kind]:
    t0 = time()
    if isEle[kind]:
        matched_idxs = gen_match(
            arrs["Ele_Gen_Phi"], arrs["phi"], arrs["Ele_Gen_Eta"], arrs["eta"]
        )
    else:
        matched_idxs = gen_match(
            arrs["Pho_Gen_Phi"], arrs["phi"], arrs["Pho_Gen_Eta"], arrs["eta"]
        )
    gen_idxs = matched_idxs[:, :, "0"]
    reco_idxs = matched_idxs[:, :, "1"]
    print("\tgen matching took %f seconds" % (time() - t0))

    t0 = time()

    for var in gen:
        arrs[var] = arrs[var][gen_idxs]

    for var in reco:
        print(var, arrs[var].type)
        arrs[var] = arrs[var][reco_idxs]

    print("\tapplying gen matching took %f seconds" % (time() - t0))

t0 = time()

# it can happen that there is exactly 1 reco electron, but it is identified as Ele2
# I have no idea why, and it's super annoying, but here we are
if not isEle[kind]:
    noEle1 = ak.num(arrs["iEtaPho1"]) == 0
else:
    noEle1 = ak.num(arrs["iEtaEle1"]) == 0

if hasgen[kind]:
    Ele1 = np.logical_and(reco_idxs == 0, ~noEle1)  # recoidx==0 and there is an ele1
else:
    Ele1 = ak.local_index(arrs["phi"]) == 0

Ele2 = ~Ele1

eventEle1 = ak.any(Ele1, axis=1)
eventEle2 = ak.any(Ele2, axis=1)

for var in gen + reco:  # per-particle information, flattened
    # result[var] = ak.to_numpy(arrs[var])
    result[var] = ak.to_numpy(
        ak.concatenate((ak.flatten(arrs[var][Ele1]), ak.flatten(arrs[var][Ele2])))
    )

for var in event:  # per-event information, broadcasted and flattened
    result[var] = ak.to_numpy(
        ak.concatenate((arrs[var][eventEle1], arrs[var][eventEle2]))
    )

for var in hits:  # hit information, flattened
    # note that this stays in awkward array format, while everything else is np
    if isEle[kind]:
        nameEle1 = var + "Ele1"
        nameEle2 = var + "Ele2"
    else:
        nameEle1 = var + "Pho1"
        nameEle2 = var + "Pho2"
    if var[-1] == "_":
        name = var[:-1]
    else:
        name = var
    result[name] = ak.concatenate(
        (arrs[nameEle1][eventEle1], arrs[nameEle2][eventEle2])
    )

print("\tbroadcasting and flattening took %f seconds" % (time() - t0))

t0 = time()

eventEle1 = ak.to_numpy(eventEle1)
eventEle2 = ak.to_numpy(eventEle2)

# event idx
# usefuly mostly for troubleshooting
result["eventidx"] = np.concatenate((eventEle1.nonzero()[0], eventEle2.nonzero()[0]))

# hit subdetector
# 1: barrel 0: endcaps
result["subdet"] = np.abs(ak.to_numpy(ak.firsts(result["Hit_Z"]))) < 300

print("\tdetermening aux features took %f seconds" % (time() - t0))

t0 = time()

savecut = savecuts[kind](result)

print(arrs["Ele_Gen_Pt"])
# for var in result.keys():
#    result[var] = result[var][savecut]

print("\tapplying savecut took %f seconds" % (time() - t0))

print("Dumping...")
for var in result.keys():
    t0 = time()
    varname = var
    with open("%s/%s.pickle" % (outfolder, varname), "wb") as f:
        pickle.dump(result[var], f, protocol=4)
    print("\tDumping %s took %f seconds" % (varname, time() - t0))

t0 = time()

print("Building target(True E)")
LeadElectronP4 = TLorentzVector()
SubLeadElectronP4 = TLorentzVector()

LeadElectronP4.SetPtEtaPhiE(
    arrs["Ele_Gen_Pt"][0],
    arrs["Ele_Gen_Eta"][0],
    arrs["Ele_Gen_Phi"][0],
    arrs["Ele_Gen_E"][0],
)

SubLeadElectronP4.SetPtEtaPhiE(
    result["Ele_Gen_Pt"][0],
    result["Ele_Gen_Eta"][0],
    result["Ele_Gen_Phi"][0],
    result["Ele_Gen_E"][0],
)
result["target"] = (LeadElectronP4 + SubLeadElectronP4).M()
print("\tBuilding target took %f seconds" % (time() - t0))

t0 = time()
print("Building cartesian features..")
cf = cartfeat(
    result["Hit_X"],
    result["Hit_Y"],
    result["Hit_Z"],
    result["RecHitEn"],
    result["RecHitFrac"],
)
print("\tBuilding features took %f seconds" % (time() - t0))
t0 = time()
result["cartfeat"] = torchify(cf)
print("\tTorchifying took %f seconds" % (time() - t0))
t0 = time()
with open("%s/cartfeat.pickle" % (outfolder), "wb") as f:
    torch.save(result["cartfeat"], f, pickle_protocol=4)
print("\tDumping took %f seconds" % (time() - t0))

print("Building projective features..")
pf = projfeat(
    result["Hit_Eta"],
    result["Hit_Phi"],
    result["Hit_Z"],
    result["RecHitEn"],
    result["RecHitFrac"],
)
print("\tBuilding features took %f seconds" % (time() - t0))
t0 = time()
result["projfeat"] = torchify(pf)
print("\tTorchifying took %f seconds" % (time() - t0))
t0 = time()
with open("%s/projfeat.pickle" % (outfolder), "wb") as f:
    torch.save(result["projfeat"], f, pickle_protocol=4)
print("\tDumping took %f seconds" % (time() - t0))

print("Building local features..")
lf = localfeat(
    result["iEta"],
    result["iPhi"],
    result["Hit_Z"],
    result["RecHitEn"],
    result["RecHitFrac"],
)
print("\tBuilding features took %f seconds" % (time() - t0))
t0 = time()
result["localfeat"] = torchify(lf)
print("\tTorchifying took %f seconds" % (time() - t0))
t0 = time()
with open("%s/localfeat.pickle" % (outfolder), "wb") as f:
    torch.save(result["localfeat"], f, pickle_protocol=4)
print("\tDumping took %f seconds" % (time() - t0))

t0 = time()
with open("trueE_target.pickle", "wb") as outpickle:
    pickle.dump(result["target"], outpickle)
print("\tDumping target took %f seconds" % (time() - t0))
print()
