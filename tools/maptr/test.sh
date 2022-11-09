python -m torch.distributed.launch --nproc_per_node=1 \
        tools/hdmaptr/test.py \
        projects/configs/hdmaptr/v3_hdmaptr_tiny_mini.py \
        work_dirs/v3_hdmaptr_tiny_mini/epoch_65.pth \
        --launcher pytorch  --eval bbox