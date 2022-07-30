#!/bin/bash

#SBATCH --partition=batch
#SBATCH --mail-user=eslam.abdelrahman@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH -J Nips_LAR_1_Imgaug_128_Nr3d_2048_cuda2_8gpus
#SBATCH -o Nips_LAR_1_Imgaug_128_Nr3d_2048_cuda2_8gpus.out
#SBATCH --nodes=1
#SBATCH --time=1-12:30:00
#SBATCH --mem=500G
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=1
#SBATCH --constraint=[v100]

#run the application:

module load cuda/10.1.243
module load gcc/6.4.0
cd /home/abdelrem/3d_codes/LAR-Look-Around-and-Refer/
cd referit3d/models/backbone/visual_encoder/pointnet2/
/home/abdelrem/anaconda3/envs/refer3d/bin/python -u setup.py install

cd /home/abdelrem/3d_codes/LAR-Look-Around-and-Refer/
/home/abdelrem/anaconda3/envs/refer3d/bin/python -u train_referit3d.py -scannet-file /ibex/scratch/abdelrem/scannet_dataset/keep_all_points_00_view_no_global_scan_alignment_densePCLoaded_saveJPG_cocoon_twoStreams_GEO/keep_all_points_00_view_no_global_scan_alignment_densePCLoaded_saveJPG_cocoon_twoStreams_GEO \
 -referit3D-file /ibex/scratch/abdelrem/scannet_dataset/nr3d.csv \
 --n-workers 8  --batch-size 10 --max-train-epochs 100  --patience 100 -load-dense True -load-imgs True --img-encoder True --imgsize 128 --object-encoder convnext_p++ --train-vis-enc-only False --experiment-tag Nips_LAR_1_Imgaug_128_Nr3d_2048_cuda2 --log-dir ./logs --cocoon False --twoStreams True --dist-url 'tcp://127.0.0.1:47891' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --obj-cls-alpha 5 --unit-sphere-norm True --feat2d ROIGeoclspred --context_2d unaligned --mmt_mask train2d --warmup --transformer --model mmt_referIt3DNet  --twoTrans True --sharetwoTrans False --tripleloss False --init-lr 0.0001 --feat2ddim 2048 --contrastiveloss False --imgaug True --camaug False  --geo3d False --clspred3d False
