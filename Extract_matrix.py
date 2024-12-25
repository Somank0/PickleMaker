import uproot
import math
import numpy as np
import matplotlib.pyplot as plt
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

Eta_Min = -1.4
Eta_Max = 1.4

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
ECAL_Max = 100

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
                E2[:, :, None],
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
            "Hit_X_Pho1",
            "Hit_Y_Pho1",
            "Hit_Z_Pho1",
		"Hit_Eta_Pho1",
		"Hit_Phi_Pho1",
		"Hit_iEta_Pho1",
		"Hit_iPhi_Pho1",
            "RecHitEnPho1",
            "RecHitEZPho1",
            "RecHitETPho1",
            "Hit_X_Pho2",
            "Hit_Y_Pho2",
            "Hit_Z_Pho2",
		"Hit_Eta_Pho2",
		"Hit_Phi_Pho2",
		"Hit_iEta_Pho2",
		"Hit_iPhi_Pho2",
            "RecHitEnPho2",
            "RecHitEZPho2",
            "RecHitETPho2",
            "m_gen",
            "p_gen",
            "pt_gen",
            "eta_gen",
            "phi_gen",
	    "SigIEIEPho1",
            "SigIPhiIPhiPho1",
            "R9_Pho1", 
	    "SigIEIEPho2",
            "SigIPhiIPhiPho2",
            "R9_Pho2", 
		#"EEFlag1",
		#"EEFlag2",
        ]

        arrs = self.tree.arrays(varnames)
        """ HISTMAKER CODE
        arrs2 = self.tree2.arrays(varnames)
        END HISTMAKER CODE """
        # Make sure that there are atleast 2 Phoctrons at gen level
        #arrs = arrs[[m< 1 for m in arrs["m_gen"]]]
        # arrs = arrs[[j[0] > 10 and j[1] > 10 for j in arrs["Gen_Pt"]]]
        # arrs = arrs[
        #    [abs(j[0]) < 1.4442 and abs(j[1]) < 1.4442 for j in arrs["Gen_Eta"]]
        # ]
        # Require at least 1 RecHit
        # arrs["Hit_X_Pho1"] = arrs["Hit_X_Pho1"][[j > 0.1 for j in arrs["RecHitEnPho1"]]]
        # arrs["Hit_Y_Pho1"] = arrs["Hit_Y_Pho1"][[j > 0.1 for j in arrs["RecHitEnPho1"]]]
        # arrs["Hit_Z_Pho1"] = arrs["Hit_Z_Pho1"][[j > 0.1 for j in arrs["RecHitEnPho1"]]]
        # arrs["RecHitEnPho1"] = arrs["RecHitEnPho1"][
        #    [j > 0.1 for j in arrs["RecHitEnPho1"]]
        # ]
        # print([[j for j in arrs["RecHitEnPho2"]]])
        # print(len([[j > 0.0 for j in arrs["Hit_Y_Pho2"]]]))
        # arrs["Hit_X_Pho2"] = arrs["Hit_X_Pho2"][[j > 0.0 for j in arrs["RecHitEnPho2"]]]
        # arrs["Hit_Y_Pho2"] = arrs["Hit_Y_Pho2"][[j > 0.0 for j in arrs["RecHitEnPho2"]]]
        # arrs["Hit_Z_Pho2"] = arrs["Hit_Z_Pho2"][[j > 0.0 for j in arrs["RecHitEnPho2"]]]
        # arrs["RecHitEnPho2"] = arrs["RecHitEnPho2"][
        #    [j > 0.1 for j in arrs["RecHitEnPho2"]]
        # ]
        arrs = arrs[[len(j) > 0 for j in arrs["Hit_X_Pho1"]]]
        """ HISTMAKER CODE
        arrs2 = arrs2[[len(j) > 0 for j in arrs2["Hit_X_Pho1"]]]
        END HISTMAKER CODE """
        # arrs = arrs[[len(j) > 0 for j in arrs["Hit_X_Pho2"]]]
        result = {}
        t0 = time()
        # angleScaled = 10000 * arrs["angle_gen"]
        # angleScaled = angleScaled + 15 * np.mean(angleScaled)
        # EBMScaled = (
        #    10000 * arrs["EBM_gen"]
        # )  # Just a multiplication factor to prevent precision loss
        # EBMScaled = EBMScaled + 15 * np.mean(
        #    EBMScaled
        # )  # Shift EBM sample corresponding to each mass
        # with open(
        #    "%s/trueAngleScaled_target.pickle" % (self.outfolder), "wb"
        # ) as outpickle:
        #    pickle.dump(angleScaled, outpickle)
        # with open(
        #    "%s/trueEBMScaled_target.pickle" % (self.outfolder), "wb"
        # ) as outpickle:
        #    pickle.dump(EBMScaled, outpickle)
        # peaks, _ = np.histogram(angleScaled)
        # with open("%s/peakinv_weights.pickle" % (self.outfolder), "wb") as outpickle:
        #    pickle.dump([1000 / np.max(peaks)] * len(EBMScaled), outpickle)
        print("\tDumping target took %f seconds" % (time() - t0))
        print("Building cartesian features..")
        """ HISTMAKER CODE
        plt.hist(arrs["m_gen"],bins=50 ,density=True, histtype='step',label="S1")
        plt.hist(arrs2["m_gen"],bins=50,density=True,linestyle='dashed' , histtype='step',label="S2")
        plt.title("M_gen before scaling")
        plt.legend()
        plt.savefig("M_gen_prescale.pdf")
        plt.savefig("M_gen_prescale.png")
        plt.clf()

        plt.hist(arrs["p_gen"],bins=50 ,density=True, histtype='step',label="S1")
        plt.hist(arrs2["p_gen"],bins=50 ,density=True, histtype='step',label="S2",linestyle='dashed')
        plt.title("P_gen before scaling")
        plt.legend()
        plt.savefig("P_gen_prescale.pdf")
        plt.savefig("P_gen_prescale.png")
        plt.clf()

        plt.hist(arrs["pt_gen"],bins=50 ,density=True, histtype='step',label="S1")
        plt.hist(arrs2["pt_gen"],bins=50 ,density=True, histtype='step',label="S2",linestyle='dashed')
        plt.title("Pt_gen before scaling")
        plt.legend()
        plt.savefig("Pt_gen_prescale.pdf")
        plt.savefig("Pt_gen_prescale.png")
        plt.clf()

        plt.hist(arrs["eta_gen"],bins=50,density=True , histtype='step',label="S1:Gen level eta of A")
        plt.hist(ak.flatten(arrs["Hit_Eta_Pho1"]),bins=50,density=True , histtype='step',label="S1:Rechit Eta of pho1")
        plt.hist(ak.flatten(arrs["Hit_Eta_Pho2"]),bins=50,density=True , histtype='step',label="S1:Rechit Eta of pho2")
        plt.hist(arrs2["eta_gen"],bins=50,density=True , histtype='step',label="S2:Gen level eta of A",linestyle='dashed')
        plt.hist(ak.flatten(arrs2["Hit_Eta_Pho1"]),bins=50,density=True , histtype='step',label="S2:Rechit Eta of pho1",linestyle='dashed')
        plt.hist(ak.flatten(arrs2["Hit_Eta_Pho2"]),bins=50,density=True , histtype='step',label="S2:Rechit Eta of pho2",linestyle='dashed')
        plt.title("Eta distributions before scaling")
        plt.legend()
        plt.savefig("Eta_prescale.pdf")
        plt.savefig("Eta_prescale.png")
        plt.clf()

        plt.hist(arrs["phi_gen"],bins=50,density=True , histtype='step',label="S1:Gen level phi of A")
        plt.hist(ak.flatten(arrs["Hit_Phi_Pho1"]),bins=50,density=True , histtype='step',label="S1:Rechit Phi of pho1")
        plt.hist(ak.flatten(arrs["Hit_Phi_Pho2"]),bins=50,density=True , histtype='step',label="S1:Rechit Phi of pho2")
        plt.hist(arrs2["phi_gen"],bins=50,density=True , histtype='step',label="S2:Gen level phi of A",linestyle='dashed')
        plt.hist(ak.flatten(arrs2["Hit_Phi_Pho1"]),bins=50,density=True , histtype='step',label="S2:Rechit Phi of pho1",linestyle='dashed')
        plt.hist(ak.flatten(arrs2["Hit_Phi_Pho2"]),bins=50,density=True , histtype='step',label="S2:Rechit Phi of pho2",linestyle='dashed')
        plt.title("Phi distributions before scaling")
        plt.legend()
        plt.savefig("Phi_prescale.pdf")
        plt.savefig("Phi_prescale.png")
        plt.clf()

        plt.hist(ak.flatten(arrs["RecHitEnPho1"]),log=True,bins=50,density=True , histtype='step',label="S1:Rechit E of pho1 before scaling")
        plt.hist(ak.flatten(arrs["RecHitEnPho2"]),log=True,bins=50,density=True , histtype='step',label="S1:Rechit E of pho2 before scaling")
        plt.hist(ak.flatten(arrs2["RecHitEnPho1"]),log=True,bins=50,density=True , histtype='step',label="S2:Rechit E of pho1 before scaling",linestyle='dashed')
        plt.hist(ak.flatten(arrs2["RecHitEnPho2"]),log=True,bins=50,density=True , histtype='step',label="S2:Rechit E of pho2 before scaling",linestyle='dashed')
        plt.title("RecHit energy distributions before scaling")
        plt.legend()
        plt.savefig("RecHitEn_prescale.pdf")
        plt.savefig("RecHitEn_prescale.png")
        plt.clf()

        m_gen=rescale(arrs["m_gen"],0,1)
        p_gen=rescale(arrs["p_gen"],0,810)
        pt_gen=rescale(arrs["pt_gen"],20,100)
        phi_gen=rescale(arrs["phi_gen"],Phi_Min,Phi_Max)
        eta_gen=rescale(arrs["eta_gen"],Eta_Min,Eta_Max)

        m_gen2=rescale(arrs2["m_gen"],0,1)
        p_gen2=rescale(arrs2["p_gen"],0,810)
        pt_gen2=rescale(arrs2["pt_gen"],20,100)
        phi_gen2=rescale(arrs2["phi_gen"],Phi_Min,Phi_Max)
        eta_gen2=rescale(arrs2["eta_gen"],Eta_Min,Eta_Max)
        END HISTMAKER CODE """

        #target=ak.Array([[m_gen[i],p_gen[i]] for i in range(len(m_gen))])
        target = arrs["m_gen"]
        #target = np.log(np.abs(target))
        """ HISTMAKER CODE
        target2 = arrs2["m_gen"]
        target2 = np.log(np.abs(target2))
        END HISTMAKER CODE """
        with open("%s/trueE_target.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(target, outpickle)
        with open("%s/p_gen.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(arrs["p_gen"], outpickle)
        with open("%s/pt_gen.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(arrs["pt_gen"], outpickle)
        with open("%s/eta_gen.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(arrs["eta_gen"], outpickle)
        with open("%s/phi_gen.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(arrs["phi_gen"], outpickle)
        with open("%s/sigieiepho1.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(arrs["SigIEIEPho1"], outpickle)
        with open("%s/sigiphiiphipho1.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(arrs["SigIPhiIPhiPho1"], outpickle)
        with open("%s/r9pho1.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(arrs["R9_Pho1"], outpickle)
        with open("%s/sigieiepho2.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(arrs["SigIEIEPho2"], outpickle)
        with open("%s/sigiphiiphipho2.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(arrs["SigIPhiIPhiPho2"], outpickle)
        with open("%s/r9pho2.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(arrs["R9_Pho2"], outpickle)
        print("Dumped variables")
        HitsX = ak.concatenate((arrs["Hit_X_Pho1"], arrs["Hit_X_Pho2"]), axis=1)
        HitsY = ak.concatenate((arrs["Hit_Y_Pho1"], arrs["Hit_Y_Pho2"]), axis=1)
        HitsZ = ak.concatenate((arrs["Hit_Z_Pho1"], arrs["Hit_Z_Pho2"]), axis=1)
        HitsEta = ak.concatenate((arrs["Hit_Eta_Pho1"], arrs["Hit_Eta_Pho2"]), axis=1)
        HitsPhi = ak.concatenate((arrs["Hit_Phi_Pho1"], arrs["Hit_Phi_Pho2"]), axis=1)
        HitsiEta = ak.concatenate((arrs["Hit_iEta_Pho1"], arrs["Hit_iEta_Pho2"]), axis=1)
        HitsiPhi = ak.concatenate((arrs["Hit_iPhi_Pho1"], arrs["Hit_iPhi_Pho2"]), axis=1)
        HitsEn = ak.concatenate((arrs["RecHitEnPho1"], arrs["RecHitEnPho2"]), axis=1)
        HitsET = ak.concatenate((arrs["RecHitETPho1"], arrs["RecHitETPho2"]), axis=1)
        HitsEZ = ak.concatenate((arrs["RecHitEZPho1"], arrs["RecHitEZPho2"]), axis=1)
        #Apply 100 MeV cut
        #HitsX = HitsX[[j > 0.01 for j in HitsEn]]
        #HitsY = HitsY[[j > 0.01 for j in HitsEn]]
        #HitsZ = HitsZ[[j > 0.01 for j in HitsEn]]
        #HitsEta = HitsEta[[j > 0.01 for j in HitsEn]]
        #HitsPhi = HitsPhi[[j > 0.01 for j in HitsEn]]
        #HitsiEta = HitsiEta[[j > 0.01 for j in HitsEn]
        #HitsiPhi = HitsiPhi[[j > 0.01 for j in HitsEn]]
        #HitsEn = HitsEn[[j > 0.01 for j in HitsEn]]
        ##HitsEn = np.log(np.abs(HitsEn))
        #HitsET = HitsET[[j > 0.01 for j in HitsEn]]
        #HitsEZ = HitsEZ[[j > 0.01 for j in HitsEn]]

        """ HISTMAKER CODE
        HitsX2 = ak.concatenate((arrs2["Hit_X_Pho1"], arrs2["Hit_X_Pho2"]), axis=1)
        HitsY2 = ak.concatenate((arrs2["Hit_Y_Pho1"], arrs2["Hit_Y_Pho2"]), axis=1)
        HitsZ2 = ak.concatenate((arrs2["Hit_Z_Pho1"], arrs2["Hit_Z_Pho2"]), axis=1)
        HitsEta2 = ak.concatenate((arrs2["Hit_Eta_Pho1"], arrs2["Hit_Eta_Pho2"]), axis=1)
        HitsPhi2 = ak.concatenate((arrs2["Hit_Phi_Pho1"], arrs2["Hit_Phi_Pho2"]), axis=1)
        HitsiEta2 = ak.concatenate((arrs2["Hit_iEta_Pho1"], arrs2["Hit_iEta_Pho2"]), axis=1)
        HitsiPhi2 = ak.concatenate((arrs2["Hit_iPhi_Pho1"], arrs2["Hit_iPhi_Pho2"]), axis=1)
        HitsEn2 = ak.concatenate((arrs2["RecHitEnPho1"], arrs2["RecHitEnPho2"]), axis=1)
        HitsET2 = ak.concatenate((arrs2["RecHitETPho1"], arrs2["RecHitETPho2"]), axis=1)
        HitsEZ2 = ak.concatenate((arrs2["RecHitEZPho1"], arrs2["RecHitEZPho2"]), axis=1)
        HitsX2 = HitsX2[[j > 0.1 for j in HitsEn2]]
        HitsY2 = HitsY2[[j > 0.1 for j in HitsEn2]]
        HitsZ2 = HitsZ2[[j > 0.1 for j in HitsEn2]]
        HitsEta2 = HitsEta2[[j > 0.1 for j in HitsEn2]]
        HitsPhi2 = HitsPhi2[[j > 0.1 for j in HitsEn2]]
        HitsiEta2 = HitsiEta2[[j > 0.1 for j in HitsEn2]]
        HitsiPhi2 = HitsiPhi2[[j > 0.1 for j in HitsEn2]]
        HitsEn2 = HitsEn2[[j > 0.1 for j in HitsEn2]]
        #HitsEn = np.log(np.abs(HitsEn))
        HitsET2 = HitsET2[[j > 0.1 for j in HitsEn2]]
        HitsEZ2 = HitsEZ2[[j > 0.1 for j in HitsEn2]]
        END HISTMAKER CODE """

        #print(HitsX, HitsY, HitsZ, HitsEn)
        #totalRechitEnergies = ak.Array([np.sum(i) for i in HitsEn])
        #with open(
        #    "%s/totalRechitEnergies.pickle" % (self.outfolder), "wb"
        #) as outpickle:
        #    pickle.dump(totalRechitEnergies, outpickle)

        Pho1flg = arrs["Hit_X_Pho1"] * 0 + 1
        Pho2flg = arrs["Hit_X_Pho2"] * 0
        """
        EEflg = np.zeros((len(arrs["Hit_X_Pho1"])))
        # Go through number of events
	#EEFLAG
        for i in range(len(arrs["Hit_X_Pho1"])):
                # If sublead photon rechits are present both need to come from endcap else only the lead photon rechits need to be from endcap
                if(len(arrs["Hit_X_Pho2"][i])>0):
                        EEflg[i] = int(arrs["EEFlag1"][i] * arrs["EEFlag2"][i])
                else:
                        EEflg[i] = int(arrs["EEFlag1"][i])
        """
        
        #Phoflags = ak.concatenate((Pho1flg, Pho2flg), axis=1)

        cf = cartfeat(
        	HitsX,
        	HitsY,
        	HitsZ,
            HitsEn,
        )
        lf = localfeat( HitsEta,HitsPhi,HitsEn)
        df = detfeat( HitsiEta,HitsiPhi,HitsEn)

        """ HISTMAKER CODE
        e = ak.flatten(rescale(HitsEta,Eta_Min,Eta_Max))
        p = ak.flatten(rescale(HitsPhi,Phi_Min,Phi_Max))
        en = ak.flatten(rescale(HitsEn,ECAL_Min,ECAL_Max))

        e2 = ak.flatten(rescale(HitsEta2,Eta_Min,Eta_Max))
        p2 = ak.flatten(rescale(HitsPhi2,Phi_Min,Phi_Max))
        en2 = ak.flatten(rescale(HitsEn2,ECAL_Min,ECAL_Max))
	
        plt.hist(target,bins=50 ,density=True, histtype='step',label="S1")
        plt.hist(target2,bins=50 ,density=True, histtype='step',label="S2",linestyle='dashed')
        plt.title("M_gen after scaling")
        plt.legend()
        plt.savefig("M_gen_postscale.pdf")
        plt.savefig("M_gen_postscale.png")
        plt.clf()

        plt.hist(p_gen,bins=50 ,density=True, histtype='step',label="S1")
        plt.hist(p_gen2,bins=50 ,density=True, histtype='step',label="S2",linestyle='dashed')
        plt.title("P_gen after scaling")
        plt.legend()
        plt.savefig("P_gen_postscale.pdf")
        plt.savefig("P_gen_postscale.png")
        plt.clf()

        plt.hist(pt_gen,bins=50 ,density=True, histtype='step',label="S1")
        plt.hist(pt_gen2,bins=50 ,density=True, histtype='step',label="S2",linestyle='dashed')
        plt.title("Pt_gen after scaling")
        plt.legend()
        plt.savefig("Pt_gen_postscale.pdf")
        plt.savefig("Pt_gen_postscale.png")
        plt.clf()

        plt.hist(eta_gen,bins=50,density=True , histtype='step',label="S1:Gen level eta of A")
        plt.hist(e,bins=50,density=True , histtype='step',label="S1:Combined Eta rechits after scaling")
        plt.hist(eta_gen2,bins=50,density=True , histtype='step',label="S2:Gen level eta of A",linestyle='dashed')
        plt.hist(e2,bins=50,density=True , histtype='step',label="S2:Combined Eta rechits after scaling",linestyle='dashed')
        plt.title("Eta distributions after scaling")
        plt.legend()
        plt.savefig("Eta_postscale.pdf")
        plt.savefig("Eta_postscale.png")
        plt.clf()

        plt.hist(phi_gen,bins=50,density=True , histtype='step',label="S1:Gen level phi of A")
        plt.hist(p,bins=50,density=True , histtype='step',label="S1:Combined Phi rechits after scaling")
        plt.hist(phi_gen2,bins=50,density=True , histtype='step',label="S2:Gen level phi of A",linestyle='dashed')
        plt.hist(p2,bins=50,density=True , histtype='step',label="S2:Combined Phi rechits after scaling",linestyle='dashed')
        plt.title("Phi distributions after scaling")
        plt.legend()
        plt.savefig("Phi_postscale.pdf")
        plt.savefig("Phi_postscale.png")
        plt.clf()

        plt.hist(en,bins=50,log=True,density=True , histtype='step',label="S1")
        plt.hist(en2,bins=50,log=True,density=True , histtype='step',label="S2",linestyle='dashed')
        plt.title("Combined RecHit energy distributions after scaling")
        plt.legend()
        plt.savefig("RecHitEn_postscale.pdf")
        plt.savefig("RecHitEn_postscale.png")
        END HISTMAKER CODE """
        print("\tBuilding features took %f seconds" % (time() - t0))
        t0 = time()
        result["cartfeat"] = torchify(cf)
        result["localfeat"] = torchify(lf)
        result["detfeat"] = torchify(df)
        print("\tTorchifying took %f seconds" % (time() - t0))
        t0 = time()
        with open("%s/cartfeat.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["cartfeat"], f, pickle_protocol=4)
        with open("%s/localfeat.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["localfeat"], f, pickle_protocol=4)
        with open("%s/detfeat.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["detfeat"], f, pickle_protocol=4)
        #with open("%s/Phoflags.pickle" % (self.outfolder), "wb") as outpickle:
        #    pickle.dump(Phoflags, outpickle)
        #with open("%s/EEflags.pickle" % (self.outfolder), "wb") as outpickle:
        #    pickle.dump(EEflg, outpickle)
        print("\tDumping took %f seconds" % (time() - t0))
        return result
