#!/bin/sh

python train.py --ckpt_path ckpt/rgb_pyramid_01 --conf_cls PhotoBaseline
python train.py --ckpt_path ckpt/rgb_occ_pyramid_01 --conf_cls RGBOcc
python train.py --ckpt_path ckpt/census_occ_01 --conf_cls CensusOcc

# For DD
# python save_teacher_flow.py --ckpt_path ckpt/census_occ_01/ckpt-200000 --dataset flyingchairs
# python train.py --ckpt_path ckpt/census_occ_dd_01 --conf_cls CensusOccDD

python train.py --ckpt_path ckpt/ours_01 --conf_cls Ours
python save_teacher_flow.py --ckpt_path ckpt/ours_01/ckpt-1000 --dataset flyingchairs
python train.py --ckpt_path ckpt/ours_final_01 --conf_cls OursFinal
