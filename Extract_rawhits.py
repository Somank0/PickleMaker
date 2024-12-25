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

# Modified by Somanko
import torch.nn.functional as F

MISSING = -999999

########################################
# HGCAL Values                         #
########################################

'''
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
'''

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
ECAL_Max = 50

ES_Min =0
#ES_Max = 100
ES_Max =0.02

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
            "RawRecHit_X_Pho1",
            "RawRecHit_Y_Pho1",
            "RawRecHit_Z_Pho1",
            "RawRecHitEnPho1",
            "RawRecHit_X_Pho2",
            "RawRecHit_Y_Pho2",
            "RawRecHit_Z_Pho2",
            "RawRecHitEnPho2",

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
            "RecHitFracPho1",
            "Hit_X_Pho2",
            "Hit_Y_Pho2",
            "Hit_Z_Pho2",
            "Hit_Eta_Pho2",
            "Hit_Phi_Pho2",
            "iEtaPho2",
            "iPhiPho2",
            "RecHitEnPho2",
            "RecHitFracPho2",
            "energy",
            "phi",
            "eta",
            "rho",
            "Pho_SigIEIE",

            "Pho_R9",

	    "A_Gen_mass",
	    "A_Gen_pt",
	    "A_Gen_eta",
	    "A_Gen_phi",
	    "Pho_Gen_Pt",
            "Pho_Gen_Eta",
            "Pho_Gen_Phi",
	    "pt",
            "HitNoisePho1",
            "HitNoisePho2",
        ]

        arrs = self.tree.arrays(varnames)
        #arrs = arrs[[len(j) > 0 for j in arrs["Hit_X_Pho1"]]]
        arrs = arrs[[len(j) > 0 for j in arrs["RawRecHit_X_Pho1"]]]
        # arrs = arrs[[len(j) > 0 for j in arrs["Hit_X_Pho2"]]]
        result = {}
        t0 = time()
        print("\tDumping target took %f seconds" % (time() - t0))
        print("Building cartesian features..")

        target = arrs["A_Gen_mass"]
        pt_gen = arrs["A_Gen_pt"]
        eta_gen = arrs["A_Gen_eta"]
        phi_gen = arrs["A_Gen_phi"]
        

       # pt_reco = ak.sum(arrs["pt"],axis=-1)
        pt_reco = arrs["pt"]
        pt_pho_gen = arrs["Pho_Gen_Pt"]
        pho_gen_eta =arrs["Pho_Gen_Eta"]
        pho_gen_phi = arrs["Pho_Gen_Phi"]
        reco_energy = arrs["energy"]

        with open("%s/pho_gen_eta.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(pho_gen_eta, outpickle)
        with open("%s/pho_gen_phi.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(pho_gen_phi, outpickle)

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
        with open("%s/reco_energy.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(reco_energy, outpickle)

        pho_r9 = arrs["Pho_R9"]
        with open("%s/pho_r9.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(pho_r9, outpickle)
        print("Dumped Gen quantities")

        chunksize=1000000
        for start in range(0,len(target),chunksize):
            stop=min(start+chunksize,len(target))

            HitsX = ak.concatenate((arrs["RawRecHit_X_Pho1"][start:stop], arrs["RawRecHit_X_Pho2"][start:stop]), axis=1)
            HitsY = ak.concatenate((arrs["RawRecHit_Y_Pho1"][start:stop], arrs["RawRecHit_Y_Pho2"][start:stop]), axis=1)
            HitsZ = ak.concatenate((arrs["RawRecHit_Z_Pho1"][start:stop], arrs["RawRecHit_Z_Pho2"][start:stop]), axis=1)
            HitsEn = ak.concatenate((arrs["RawRecHitEnPho1"][start:stop], arrs["RawRecHitEnPho2"][start:stop]), axis=1)

        #Rechit_frac = ak.concatenate((arrs["RecHitFracPho1"],arrs["RecHitFracPho2"]),axis=1)

        #HitNoise= ak.concatenate((arrs["HitNoisePho1"],arrs["HitNoisePho2"]),axis=1)

        #HitsPhi = ak.concatenate((arrs["Hit_Phi_Pho1"], arrs["Hit_Phi_Pho2"]), axis=1)
        #HitsiEta = ak.concatenate((arrs["iEtaPho1"], arrs["iEtaPho2"]), axis=1)
        #HitsiPhi = ak.concatenate((arrs["iPhiPho1"], arrs["iPhiPho2"]), axis=1)

            HitsESX = ak.concatenate((arrs["Hit_ES_X_Pho1"][start:stop], arrs["Hit_ES_X_Pho2"][start:stop]), axis=1)
            HitsESY = ak.concatenate((arrs["Hit_ES_Y_Pho1"][start:stop], arrs["Hit_ES_Y_Pho2"][start:stop]), axis=1)
            HitsESZ = ak.concatenate((arrs["Hit_ES_Z_Pho1"][start:stop], arrs["Hit_ES_Z_Pho2"][start:stop]), axis=1)
            HitsESEn = ak.concatenate((arrs["ES_RecHitEnPho1"][start:stop], arrs["ES_RecHitEnPho2"][start:stop]), axis=1)
        #HitsESEta = ak.concatenate((arrs["Hit_ES_Eta_Pho1"], arrs["Hit_ES_Eta_Pho2"]), axis=1)
        #HitsESPhi = ak.concatenate((arrs["Hit_ES_Phi_Pho1"], arrs["Hit_ES_Phi_Pho2"]), axis=1)
            #HitsEE_En = HitsEn
            HitsX = rescale(HitsX,X_Min,X_Max)
            HitsY = rescale(HitsY,Y_Min,Y_Max)
            HitsZ = rescale(HitsZ,Z_Min,Z_Max)
            HitsEn = rescale(HitsEn,ECAL_Min,ECAL_Max)
        #HitsEn = HitsEn * Rechit_frac

            HitsESX = rescale(HitsESX,X_Min,X_Max)
            HitsESY =  rescale(HitsESY,Y_Min,Y_Max)
            HitsESZ = rescale(HitsESZ,Z_Min,Z_Max)
            HitsESEn = rescale(HitsESEn,ES_Min,ES_Max)
            EEflg=HitsX*0
            ESflg=HitsESX*0 +1        

        #Pho1flg = arrs["RawRecHit_X_Pho1"] * 0 
        #Pho2flg = arrs["RawRecHit_X_Pho2"] * 0
        #ESPho1flg = arrs["Hit_ES_X_Pho1"] * 0 +1
        #ESPho2flg = arrs["Hit_ES_X_Pho2"] * 0 +1
        #EEflg = ak.concatenate((Pho1flg,Pho2flg),axis=1)
        #ESflg = ak.concatenate((ESPho1flg,ESPho2flg),axis=1)
        
            HitsX = ak.concatenate((HitsX,HitsESX),axis=-1)
            HitsY = ak.concatenate((HitsY,HitsESY),axis=-1)
            HitsZ = ak.concatenate((HitsZ,HitsESZ),axis=-1)
            HitsEn = ak.concatenate((HitsEn,HitsESEn),axis=-1)
        #HitsEta=ak.concatenate((HitsEta,HitsESEta),axis=-1)
        #HitsPhi=ak.concatenate((HitsPhi,HitsESPhi),axis=-1)
            flg = ak.concatenate((EEflg,ESflg),axis=-1)
        
            print("\tLoading features took %f seconds" %(time() - t0))
            t0=time()
            cf=cartfeat(HitsX,HitsY,HitsZ,HitsEn,flg)
            result[f'cartfeat_{start}'] = torchify(cf)
            print("\tTorchifying took %f seconds"%(time() - t0))
            with open(f"%s/cartfeat_{start}.pickle" % (self.outfolder), "wb") as f:
                torch.save(result[f"cartfeat_{start}"], f, pickle_protocol=4)
            print("\tDumping took %f seconds" % (time() - t0))

        #print(eta)
        #graph_x = gx


        #cf = cartfeat(
        	#HitsX,
        	#HitsY,
        	#HitsZ,
                #HitsEn,
                #flg
        #)
        
        #cfes = cartfeat(HitsESX,HitsESY,HitsESZ, HitsESEn,ESflg)
        
        #lf = localfeat(HitsEta,HitsPhi,HitsEE_En)
        #df = detfeat(HitsiEta,HitsiPhi,HitsEE_En)

        
        '''print("\tBuilding features took %f seconds" % (time() - t0))
        t0 = time()
        #result["cartfeat_ES"] = torchify(cf,graph_x)
        result["cartfeat_ES"]= torchify(cf)
        #result["localfeat"] = torchify(lf)
        #result["detfeat"] = torchify(df)
        #result["sc_eta"] = gx


        print("\tTorchifying took %f seconds" % (time() - t0))
        t0 = time()
        #with open("%s/cartfeat.pickle" % (self.outfolder), "wb") as f:
            #torch.save(result["cartfeat"], f, pickle_protocol=4)
        with open("%s/cartfeat_ES.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["cartfeat_ES"], f, pickle_protocol=4)
        #with open("%s/sc_eta.pickle" % (self.outfolder), "wb") as f:
            #torch.save(result["sc_eta"], f, pickle_protocol=4)'''
        '''with open("%s/localfeat.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["localfeat"], f, pickle_protocol=4)
        with open("%s/detfeat.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["detfeat"], f, pickle_protocol=4)'''
        
        '''print("\tDumping took %f seconds" % (time() - t0))'''
        return result
