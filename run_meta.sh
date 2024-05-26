# #!/bin/bash


python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
            --cfg ./configs/MetaFG_1_224.yaml \
            --batch-size 32 \
            --tag fishair \
            --lr 5e-5 \
            --min-lr 5e-7 \
            --warmup-lr 5e-8 \
            --epochs 300 \
            --warmup-epochs 20 \
            --dataset fishair_processed \
            --data-path /data/DatasetTrackFinalData/Classification \
            --pretrain ./pretrained_model/metafg_1_1k_224.pth \
            --accumulation-steps 2 \
            --opts DATA.IMG_SIZE 224 \
            --amp-opt-level O0
