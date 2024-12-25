This is the Pickle maker for A to gamma gamma sample pickle files to train the DRN

The inputs given are the x,y,z, Et, Ez and subdetector flag (for endcaps and preshower) of all rechits within a dR of 0.3 from the supercluster seed.

If the pickle maker is run on NSM area (Parambrahma), first run the following commands to set up the environment.

module load iiser/apps/cuda/11.4 

module load cdac/spack/0.17
source /home/apps/spack/share/spack/setup-env.sh
spack load python@3.8.2
source /home/apps/iiser/pytorch-venv/bin/activate


To run the pickle maker 

Change the paths in the nTuple and folder variables in MypreparePickles_ES_EE as needed.

The exectuable MypreparePickles_ES_EE calls MyExtract_drhits_EtEz.py. If changes needs to be made in the inputs (like rescaling or adding new features) , it should be done in MyExtract_drhits_EtEz.py file.

To prepare the pickles, run the following command:

./MypreparePickles_ES_EE 1 1 1

This prepares pickles in chunks of 1M (to prevent the system from running out of memory).

To combine the chunks of pickle files, assign the path where the chunked pickles are stored to the variable "output_dir"  in Combine_pickle.py . If ES hits are included, change value of "output_file" to "cartfeat_ES.pickle".

Combine the chunked pickles by running the following command:

python Combine_pickle.py


