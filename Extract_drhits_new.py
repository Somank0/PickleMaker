import uproot
import math
import numpy as np
#import matplotlib.pyplot as plt
import awkward as ak
from time import time
import pickle
import tqdm
from torch_geometric.data import Data
import torch
from MyDataset import MyDataset
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

Eta_Min = -2.5
Eta_Max = 2.5

Phi_Min = -np.pi
Phi_Max = np.pi

iEta_Min = -100
iEta_Max = 120

iPhi_Min = 1
iPhi_Max = 360

iX_Min = 1
iX_Max = 100

iY_Min = 1
iY_Max = 100

ECAL_Min = 0
ECAL_Max = 100

ES_Min =0
#ES_Max = 100
ES_Max =0.1

#############################################

def localfeat(eta,phi,energy,energy2=None):
	
	e = rescale(eta,Eta_Min,Eta_Max)
	p = rescale(phi,Phi_Min,Phi_Max)
	en = rescale(energy,ECAL_Min,ECAL_Max)
	#en2 = rescale(energy2,ECAL_Min,ECAL_Max)
	return ak.concatenate(
            (
                e[:, :, None],
                p[:, :, None],
                en[:, :, None],
	#	en2[:,:,None],
            ),
            -1,
        )

def detfeat(ieta,iphi,energy,energy2=None):
	
	return ak.concatenate(
            (
                ieta[:, :, None],
                iphi[:, :, None],
                energy[:, :, None],
	#	en2[:,:,None],
            ),
            -1,
        )
	

def detfeat(ieta,iphi,energy):
	e = rescale(ieta,iEta_Min,iEta_Max)
	p = rescale(iphi,iPhi_Min,iPhi_Max)
	en = rescale(energy,ECAL_Min,ECAL_Max)
	return ak.concatenate(
            (
                e[:, :, None],
                p[:, :, None],
                en[:, :, None],
            ),
            -1,
        )

def localfeat(eta,phi,energy,det=None):
	
    e = rescale(eta,Eta_Min,Eta_Max)
    p = rescale(phi,Phi_Min,Phi_Max)
    en = rescale(energy,ECAL_Min,ECAL_Max)
	#en2 = rescale(energy2,ECAL_Min,ECAL_Max)
    if det is None :
     return ak.concatenate(
            (
                e[:, :, None],
                p[:, :, None],
                en[:, :, None],
	#	en2[:,:,None],
            ),
            -1,
        )
    else :
     return ak.concatenate(
            (
                e[:, :, None],
                p[:, :, None],
                en[:, :, None],
		det[:,:,None],
            ),
            -1,
        )

def detfeat(ieta,iphi,energy,det=None):
    if det is None :
        return ak.concatenate(
                (
                    ieta[:, :, None],
                    iphi[:, :, None],
                    energy[:, :, None],
        #	en2[:,:,None],
                ),
                -1,
           )
    else:
        return ak.concatenate(
                (
                    ieta[:, :, None],
                    iphi[:, :, None],
                    energy[:, :, None],
		    det[:,:,None],
                ),
                -1,
            )
	
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


def cartfeat(x, y, z, E, det=None):
    #E = rescale(E, ECAL_Min, ECAL_Max)
    #x = rescale(x, X_Min, X_Max)
    #y = rescale(y, Y_Min, Y_Max)
    #z = rescale(z, Z_Min, Z_Max)


    if det is None :
        return ak.concatenate(
            (x[:, :, None], y[:, :, None], z[:, :, None], E[:, :, None]),
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
        Data(x=torch.from_numpy(ak.to_numpy(Pho).astype(np.float32))) for Pho in feat
    ]
    if graph_x is not None:
        for d, gx in zip(data, graph_x):
            d.graph_x = gx
    return data


def npify(feat):
    t0 = time()
    data = [ak.to_numpy(Pho) for Pho in feat]
    print("took %f" % (time() - t0))
    return data

def make_feat_v4(x_ECAL, y_ECAL, z_ECAL, E_ECAL,
       # f_ECAL, noise_ECAL = None,
        #good_ECAL = None, time_ECAL = None, calib_ECAL = None, gain_ECAL=None,
        x_ES = None, y_ES=None, z_ES=None, E_ES=None, 
        #good_ES = None,
        rho = None, HoE=None
        ):
    print("rescaling")
    x_ECAL = rescale(x_ECAL, X_Min, X_Max)
    y_ECAL = rescale(y_ECAL, Y_Min, Y_Max)
    z_ECAL = rescale(z_ECAL, Z_Min, Z_Max)
    E_ECAL = rescale(E_ECAL , ECAL_Min, ECAL_Max)

    #if noise_ECAL is None:
    feat_ECAL = ak.concatenate( (x_ECAL[:,:,None], 
                                     y_ECAL[:,:,None], 
                                     z_ECAL[:,:,None], 
                                     E_ECAL[:,:,None]), -1)
    if x_ES is not None:
        x_ES = rescale(x_ES, X_Min, X_Max)
        y_ES = rescale(y_ES, Y_Min, Y_Max)
        z_ES = rescale(z_ES, Z_Min, Z_Max)
        E_ES = rescale(E_ES, ES_Min, ES_Max)

        feat_ES = ak.concatenate( (x_ES[:,:,None],
                                   y_ES[:,:,None],
                                   z_ES[:,:,None],
                                   E_ES[:,:,None]), -1)
    if rho is not None:
        rho = rescale(rho, Rho_Min, Rho_Max)
    if HoE is not None:
        HoE = rescale(HoE, HoE_Min, HoE_Max)

    if rho is not None and HoE is not None:
        gx = np.concatenate((rho[:,None], HoE[:,None]), -1)
    elif rho is not None:
        gx = rho
    elif HoE is not None:
        gx = HoE
    else:
        gx = None


    
    #torchify
    print("torchifying")
    if feat_ECAL is not None:
       # feat_ECAL = ak.pad_none(feat_ECAL, 5 if noise_ECAL is not None else 0, 2)
        feat_ECAL = [torch.from_numpy(ak.to_numpy(y).astype(np.float32)) for y in feat_ECAL]
    if x_ES is not None:
        feat_ES = ak.pad_none(feat_ES, 4, 2, clip=True)
        feat_ES = [torch.from_numpy(ak.to_numpy(x).astype(np.float32)) for x in feat_ES]
    else:
        feat_ES = [None] * len(feat_ECAL)
    if gx is not None:
        gx = [torch.from_numpy(np.asarray(ak.to_numpy(x)).astype(np.float32)) for x in gx]
    else:
        gx = [None] * len(feat_ECAL)

    print("building list")
    #data = [makedata(xECAL, fECAL, xES, fES, xgx) for xECAL, fECAL, xES, fES, xgx in zip(feat_ECAL, flags_ECAL, feat_ES, flags_ES, gx)]
    data = [makedata(xECAL, xES, xgx) for xECAL, xES, xgx in zip(feat_ECAL, feat_ES, gx)]


    print("done")
    return data

'''def makedata(xECAL, fECAL, xES, fES, xgx):
    result = Data()
    result.xECAL = xECAL
    result.fECAL = fECAL
    result.xES = xES
    result.fES = fES
    result.gx = xgx
    return result
'''

def makedata(xECAL, xES, xgx):
    result = Data()
    result.xECAL = xECAL
   # result.fECAL = fECAL
    result.xES = xES
   #result.fES = fES
    #result = MyDataset(xECAL, xES)
    result.num_nodes = (torch.cat((xECAL,xES),0)).size(0)
    result.gx = xgx
    return result

class Extract:
    def __init__(
        self,
        outfolder="pickles",
        path="merged.root",
        treeName="nTuplelize/T",
        path2=None,
    ):
        if path is not None:
            # path = '~/shared/nTuples/%s'%path
            self.tree = uproot.open("%s:%s" % (path, treeName))
        if path2 is not None:
            # path = '~/shared/nTuples/%s'%path
            self.tree2 = uproot.open("%s:%s" % (path2, treeName))

        self.outfolder = outfolder

    def read(self, N=None):
        varnames = [
            "Hit_ES_X_Pho1",
            "Hit_ES_Y_Pho1",
            "Hit_ES_Z_Pho1",
            "Hit_ES_Eta_Pho1",
            "Hit_ES_Phi_Pho1",
            "ES_RecHitEnPho1",
            "Hit_ES_Eta_Pho2",
            "Hit_ES_Phi_Pho2",
            "Hit_ES_X_Pho2" ,         
            "Hit_ES_Y_Pho2",
            "Hit_ES_Z_Pho2",
            "ES_RecHitEnPho2",
            "Hit_X_Pho1",
            "Hit_Y_Pho1",
            "Hit_Z_Pho1",
            "Hit_Eta_Pho1",
            "Hit_Phi_Pho1",
            "iEtaPho1",
            "iPhiPho1",
            "RecHitEnPho1",
            "Hit_X_Pho2",
            "Hit_Y_Pho2",
            "Hit_Z_Pho2",
            "Hit_Eta_Pho2",
            "Hit_Phi_Pho2",
            "iEtaPho2",
            "iPhiPho2",
            "RecHitEnPho2",
	    "A_Gen_mass",
	    "A_Gen_pt",
	    "A_Gen_eta",
	    "A_Gen_phi",
	    "Pho_Gen_Pt",
	    "pt"
        ]

        arrs = self.tree.arrays(varnames)
        arrs = arrs[[len(j) > 0 for j in arrs["Hit_X_Pho1"]]]
        # arrs = arrs[[len(j) > 0 for j in arrs["Hit_X_Pho2"]]]
        result = {}
        t0 = time()
        print("\tDumping target took %f seconds" % (time() - t0))
        print("Building cartesian features..")

        target = arrs["A_Gen_mass"]
        pt_gen = arrs["A_Gen_pt"]
        eta_gen = arrs["A_Gen_eta"]
        phi_gen = arrs["A_Gen_phi"]
        pt_reco = ak.sum(arrs["pt"],axis=-1)
        pt_pho_gen = arrs["Pho_Gen_Pt"]
        with open("%s/pt_pho_gen.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(pt_pho_gen, outpickle)
        print("Dumped Pt Pho gen")
        with open("%s/pt_reco.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(pt_reco, outpickle)
        print("Dumped Pt reco")
        with open("%s/trueE_target.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(target, outpickle)
        with open("%s/pt_gen.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(pt_gen, outpickle)
        with open("%s/eta_gen.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(eta_gen, outpickle)
        with open("%s/phi_gen.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(phi_gen, outpickle)
        with open("%s/m_pt_target.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(ak.Array([[i[0],j[0]] for i,j in zip(target,pt_gen)]), outpickle)
        with open("%s/two_reco.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump([len(j) > 0 for j in arrs["Hit_X_Pho2"]], outpickle)
        print("Dumped Gen quantities")
        HitsX = ak.concatenate((arrs["Hit_X_Pho1"], arrs["Hit_X_Pho2"]), axis=1)
        HitsY = ak.concatenate((arrs["Hit_Y_Pho1"], arrs["Hit_Y_Pho2"]), axis=1)
        HitsZ = ak.concatenate((arrs["Hit_Z_Pho1"], arrs["Hit_Z_Pho2"]), axis=1)
        HitsEn = ak.concatenate((arrs["RecHitEnPho1"], arrs["RecHitEnPho2"]), axis=1)
        HitsEta = ak.concatenate((arrs["Hit_Eta_Pho1"], arrs["Hit_Eta_Pho2"]), axis=1)
        HitsPhi = ak.concatenate((arrs["Hit_Phi_Pho1"], arrs["Hit_Phi_Pho2"]), axis=1)
        HitsiEta = ak.concatenate((arrs["iEtaPho1"], arrs["iEtaPho2"]), axis=1)
        HitsiPhi = ak.concatenate((arrs["iPhiPho1"], arrs["iPhiPho2"]), axis=1)
        HitsESX = ak.concatenate((arrs["Hit_ES_X_Pho1"], arrs["Hit_ES_X_Pho2"]), axis=1)
        HitsESY = ak.concatenate((arrs["Hit_ES_Y_Pho1"], arrs["Hit_ES_Y_Pho2"]), axis=1)
        HitsESZ = ak.concatenate((arrs["Hit_ES_Z_Pho1"], arrs["Hit_ES_Z_Pho2"]), axis=1)
        HitsESEn = ak.concatenate((arrs["ES_RecHitEnPho1"], arrs["ES_RecHitEnPho2"]), axis=1)
        HitsESEta = ak.concatenate((arrs["Hit_ES_Eta_Pho1"], arrs["Hit_ES_Eta_Pho2"]), axis=1)
        HitsESPhi = ak.concatenate((arrs["Hit_ES_Phi_Pho1"], arrs["Hit_ES_Phi_Pho2"]), axis=1)
        HitsEE_En = HitsEn
        
        #HitsX = rescale(HitsX,X_Min,X_Max)
        #HitsY = rescale(HitsY,Y_Min,Y_Max)
        #HitsZ = rescale(HitsZ,Z_Min,Z_Max)
        #HitsEn = rescale(HitsEn,ECAL_Min,ECAL_Max)

        #HitsESX = rescale(HitsESX,X_Min,X_Max)
        #HitsESY =  rescale(HitsESY,Y_Min,Y_Max)
        #HitsESZ = rescale(HitsESZ,Z_Min,Z_Max)
        #HitsESEn = rescale(HitsESEn,ES_Min,ES_Max)
        
        

        Pho1flg = arrs["Hit_X_Pho1"] * 0 
        Pho2flg = arrs["Hit_X_Pho2"] * 0
        ESPho1flg = arrs["Hit_ES_X_Pho1"] * 0 +1
        ESPho2flg = arrs["Hit_ES_X_Pho2"] * 0 +1
        EEflg = ak.concatenate((Pho1flg,Pho2flg),axis=1)
        ESflg = ak.concatenate((ESPho1flg,ESPho2flg),axis=1)
        
        #HitsX = ak.concatenate((HitsX,HitsESX),axis=-1)
        #HitsY = ak.concatenate((HitsY,HitsESY),axis=-1)
        #HitsZ = ak.concatenate((HitsZ,HitsESZ),axis=-1)
        #HitsEn = ak.concatenate((HitsEn,HitsESEn),axis=-1)
        #HitsEta=ak.concatenate((HitsEta,HitsESEta),axis=-1)
        #HitsPhi=ak.concatenate((HitsPhi,HitsESPhi),axis=-1)
        #flg = ak.concatenate((EEflg,ESflg),axis=-1)
        '''
        cf = cartfeat(
		HitsX,
		HitsY,
		HitsZ,
                HitsEn,
                EEflg
        )
        '''
        #cfes = cartfeat(HitsESX,HitsESY,HitsESZ, HitsESEn,ESflg)
        
        lf = localfeat(HitsEta,HitsPhi,HitsEn)#,EEflg)
        df = detfeat(HitsiEta,HitsiPhi,HitsEE_En)

        '''
        print("\tBuilding features took %f seconds" % (time() - t0))
        t0 = time()
        #result["cartfeat"] = torchify(cf)
        result["cartfeat_ES"]= torchify(cf)
        result["localfeat"] = torchify(lf)
        result["detfeat"] = torchify(df)


        print("\tTorchifying took %f seconds" % (time() - t0))
        t0 = time()
        #with open("%s/cartfeat.pickle" % (self.outfolder), "wb") as f:
        #    torch.save(result["cartfeat"], f, pickle_protocol=4)
        with open("%s/cartfeat_ES.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["cartfeat_ES"], f, pickle_protocol=4)

        with open("%s/localfeat.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["localfeat"], f, pickle_protocol=4)
        with open("%s/detfeat.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["detfeat"], f, pickle_protocol=4)
        '''
        feats= make_feat_v4(HitsX,HitsY,HitsZ,HitsEn, HitsESX,HitsESY,HitsESZ,HitsESEn)
        
        print("\tDumping took %f seconds" % (time() - t0))
        with open("%s/cartfeat_ES.pickle" % (self.outfolder), "wb") as f:
            torch.save(feats, f, pickle_protocol=4)
            #pickle.dump(feats,f)

        return feats
