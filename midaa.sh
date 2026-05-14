
#!/bin/sh
#BSUB -J MIDAA
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -W 6:00
#BSUB -o logs/midaa_%J.out
#BSUB -e logs/midaa_%J.err
#BSUB -B
#BSUB -N

module load python3/3.11.3
module load cuda/12.4
source ~/deep-learning/venv_midaa/bin/activate

mkdir -p results logs

echo "=== PyTorch GPU check ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== MNIST ==="
python MIDAA.py \
    --dataset mnist \
    --subset 1 \
    --n_arc_min 4 \
    --n_arc_max 20 \
    --n_runs 4 \
    --R 5 \
    --steps 2500 \
    --decoder_epochs 50 \
    --outdir results

echo "=== Blood ==="
python MIDAA.py \
    --dataset blood \
    --subset 1 \
    --n_arc_min 4 \
    --n_arc_max 20 \
    --n_runs 4 \
    --R 5 \
    --steps 2500 \
    --decoder_epochs 50 \
    --outdir results