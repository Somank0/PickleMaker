#!/bin/bash -l
#SBATCH --time=72:00:00
#SBATCH --ntasks=4
#SBATCH --mem=100g
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mail-user=somanko.saha@students.iiserpune.ac.in
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --job-name="timAhcal_valid"

folder="/home/sosaha/test12_model3"
#folder="/home/sosaha/test11"

#v2->trueE, no semi
#v1->semi_dscb, loss fun= dscb

#export PYTHONUNBUFFERED=1
#module load ohpc
#module load iiser/apps/cuda/11.4
#module load cmake/3.14.3
#module swap gnu8 cdac/compiler/gcc/10.2.0
#module load python/3.9.8
module load iiser/apps/cuda/11.4 

module load cdac/spack/0.17
source /home/apps/spack/share/spack/setup-env.sh
spack load python@3.8.2
source /home/apps/iiser/pytorch-venv/bin/activate

#module load cdac/compiler/gcc/10.2.0
#module spider graphviz
#source activate torch1.8
/home/sosaha/DRN/The_DRN_for_AToGG/train $folder /home/sosaha/PickleMaker/Extractor_for_Merged/1M_SP_test2 --nosemi --idx_name all --target trueE --in_layers 3 --mp_layers 4 --out_layers 2  --agg_layers 3 --valid_batch_size 300 --train_batch_size 300  --lr_sched Const --max_lr 0.0001 --pool max --ES yes --hidden_dim 64 --n_epochs 100 --warm /home/sosaha/test12_model3/checkpoints_old/model_checkpoint_DataParallel_186310_a04c4d6531_sosaha.best.pth.tar &>> $folder/training.log
