
#!/bin/sh
#BSUB -J MIDAA_test
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 0:30
#BSUB -o logs/midaa_test_%J.out
#BSUB -e logs/midaa_test_%J.err
#BSUB -N

module load python3/3.11.3
module load cuda/12.4
source ~/deep-learning/venv_midaa/bin/activate

mkdir -p results_test logs

echo "=== PyTorch GPU check ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== MNIST test ==="
python MIDAA.py \
    --dataset mnist \
    --subset 0.02 \
    --n_arc_min 6 \
    --n_arc_max 8 \
    --n_runs 2 \
    --R 2 \
    --steps 50 \
    --decoder_epochs 5 \
    --outdir results_test

echo "=== Blood test ==="
python MIDAA.py \
    --dataset blood \
    --subset 0.02 \
    --n_arc_min 6 \
    --n_arc_max 8 \
    --n_runs 2 \
    --R 2 \
    --steps 50 \
    --decoder_epochs 5 \
    --outdir results_test

echo "=== Checking output files ==="
python -c "
import torch
for f in ['results_test/midaa_mnist_results.pt', 'results_test/midaa_blood_results.pt']:
    r = torch.load(f, weights_only=False)
    print(f, '-> keys:', list(r.keys()))
"