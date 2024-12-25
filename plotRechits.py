import uproot
from ROOT import TFile
from ROOT import TH1F
from ROOT import TCanvas

tree = uproot.open("fordrn_total.root:fordrn")
data = tree.arrays(["Hit_X_Ele2", "Hit_X_Ele1", "event"])
hits1 = data["Hit_X_Ele1"]
hits_hist1 = TH1F("hits_hist", "Number of Rechits for lead electron", 200, 0, 100)
for i in hits1:
    hits_hist1.Fill(len(i))
hits2 = data["Hit_X_Ele2"]
hits_hist2 = TH1F("hits_hist", "Number of Rechits for sublead electron", 200, 0, 100)
for i in hits2:
    hits_hist2.Fill(len(i))
canvas = TCanvas("Rechits_Plot", "Rechits_Plot", 800, 700)
canvas.Divide(2, 1)
canvas.cd(1)
hits_hist1.Draw("HIST")
canvas.cd(2)
hits_hist2.Draw("HIST")
canvas.Modified()
canvas.Update()
canvas.Write()
canvas.SaveAs("rechitsdist.root")
