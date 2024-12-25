import os
import pickle
import awkward as ak
from time import time
import torch
#output_dir="Final_5M_EE_ES_pickles"
#output_dir="1M_74K_test_noise_pickle"
#output_file="cartfeat_ES.pickle"
output_file="cartfeat.pickle"
#output_dir="1M_AtoGG_Et_Ez_pickles"
#output_dir="EB_4M_rawhits_pickle"
#output_dir="EB_4M_supcl_dr_pickles"
output_dir="EB_test1_raw_EtEz"
all_data=[]
t0=time()
pickle_files = [f for f in os.listdir(output_dir) if f.startswith("cartfeat_") and f.endswith(".pickle")]
sorted_files = sorted(pickle_files, key=lambda f: int(f.split('_')[1].split('.pickle')[0]))
for filename in sorted_files:
    if filename.startswith("cartfeat_") and filename.endswith(".pickle"):
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "rb") as f:
            data = torch.load(f)
        all_data = all_data + data
        print(f"Filename : {filename}")
        os.remove(file_path)
print("\tConcatenating took %f seconds" %(time()-t0))
t0=time()
output_path=os.path.join(output_dir,output_file)
with open(output_path,"wb") as f:
    torch.save(all_data, f, pickle_protocol=4) 
print("\tDumping features took %f seconds"%(time()-t0))
