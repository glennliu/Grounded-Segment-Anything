#!/usr/bin/env bash

SPLIT=$1
SPLIT_NAME=$2

python automatic_label_ram_demo.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --ram_checkpoint checkpoints/ram_swin_large_14m.pth --grounded_checkpoint checkpoints/groundingdino_swint_ogc.pth --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth --dataroot /data2/bim --dataset bim --split test --split_file test --frame_gap 1 --box_threshold 0.35 --text_threshold 0.25 --iou_threshold 0.3 --device cuda --augment_off
