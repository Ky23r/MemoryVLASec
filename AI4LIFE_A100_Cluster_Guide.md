# NVIDIA DGX A100 Cluster User Guide

## 1. SSH Access

Ensure that your SSH public key has been registered and that your account has been granted access to the cluster.

Add the following configuration to your local SSH configuration file.

### Linux / macOS

```text
~/.ssh/config
```

### Windows

```text
C:\Users\<username>\.ssh\config
```

Example:

```ssh
Host server_a100_user03_proxy
    ProxyJump aiotlab_92_proxy
    HostName localhost
    User user03
    Port 1112

Host aiotlab_92_proxy
    HostName 222.252.4.41
    User aiotlab_proxy
    Port 1111
```

Update the username, port, or host information according to the credentials assigned to your account.

Connect to the cluster:

```bash
ssh server_a100_user03_proxy
```

The same SSH host can be used with **VS Code Remote - SSH**.

---

## 2. Python Environment

Miniconda is already installed on the cluster.

Create and activate a Conda environment:

```bash
conda create -n myenv python=3.10 -y
conda activate myenv
```

The shared `/tmp` filesystem has limited capacity. Large package installations may fail with:

```text
OSError(28, 'No space left on device')
```

Use a temporary directory under your home directory:

```bash
export TMPDIR=~/.pip_tmp
mkdir -p "$TMPDIR"
```

Install PyTorch with the CUDA 12.6 build:

```bash
pip install torch \
    --index-url https://download.pytorch.org/whl/cu126
```

For projects with a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Additional packages can be installed normally:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Running:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

directly on the head node will normally return:

```text
False
```

This is expected because GPU resources are available only inside jobs allocated by SLURM.

---

## 3. SLURM Job Submission

Do not run compute-intensive workloads directly on the head node. GPU workloads must be submitted through SLURM.

A typical project structure is:

```text
~/my_project/
├── train.py
├── requirements.txt
├── data/
├── output/
├── logs/
└── run_job.sh
```

Create the log and output directories before submitting a job:

```bash
mkdir -p ~/my_project/logs ~/my_project/output
```

SLURM opens the files specified by `--output` and `--error` before executing the script, so the `logs/` directory must already exist.

Create a SLURM script:

```bash
nano ~/my_project/run_job.sh
```

Example:

```bash
#!/bin/bash

#SBATCH --job-name=my_train_job
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -e

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

cd ~/my_project

python train.py \
    --epochs 100 \
    --batch-size 32 \
    --output output/

echo "Job completed: $(date)"
```

Relevant SLURM options:

| Option               | Description              |
| -------------------- | ------------------------ |
| `--partition=defq`   | GPU partition            |
| `--qos=normal`       | Job Quality of Service   |
| `--gres=gpu:1`       | Number of requested GPUs |
| `--cpus-per-task=16` | Number of CPU cores      |
| `--mem=128G`         | Requested system memory  |
| `--time=24:00:00`    | Maximum runtime          |

Available QoS options include:

```text
normal
short
```

Use `short` for jobs that complete within 24 hours when appropriate.

The line:

```bash
set -e
```

causes the script to stop when a command fails, preventing later successful shell commands from hiding an earlier Python error.

Submit the job:

```bash
cd ~/my_project
sbatch run_job.sh
```

A successful submission returns:

```text
Submitted batch job 12345
```

---

## 4. Job Monitoring

Display all jobs owned by the current user:

```bash
squeue --me
```

Inspect a specific job:

```bash
squeue -j <job_id>
```

Common states:

| State | Meaning    |
| ----- | ---------- |
| `PD`  | Pending    |
| `R`   | Running    |
| `CG`  | Completing |
| `F`   | Failed     |
| `CA`  | Cancelled  |

If a job remains in `PD`, check the `NODELIST(REASON)` column.

Common reasons include:

```text
AssocMaxGRESPerJob
```

The requested GPU allocation exceeds the permitted limit.

```text
QOSMaxWallDurationPerJobLimit
```

The requested runtime exceeds the selected QoS limit.

If the job submission limit is exceeded, SLURM may return:

```text
sbatch: error: AssocMaxSubmitJobLimit
```

Cancel a job with:

```bash
scancel <job_id>
```

Cancel all jobs owned by the current user:

```bash
scancel --me
```

---

## 5. Job Logs and Completion Status

Display standard output:

```bash
cat logs/output_<job_id>.log
```

Follow the output in real time:

```bash
tail -f logs/output_<job_id>.log
```

Display errors:

```bash
cat logs/error_<job_id>.log
```

After the job finishes, check its final SLURM status:

```bash
sacct -j <job_id> \
    --format=JobID,JobName,State,ExitCode,Elapsed,AllocGRES
```

Example:

```text
       JobID    JobName      State ExitCode    Elapsed  AllocGRES
------------ ---------- ---------- -------- ---------- ----------
       12345  my_train_  COMPLETED      0:0   02:34:15      gpu:1
```

A normally completed job should report:

```text
State = COMPLETED
ExitCode = 0:0
```

Application logs should still be checked to confirm that the program produced the expected results.

---

## 6. GPU Validation

GPU information should be checked from inside a SLURM job that has been allocated GPU resources.

Run:

```bash
nvidia-smi
```

Check GPU availability from PyTorch:

```bash
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
"
```

Expected output for an allocated A100 GPU:

```text
CUDA available: True
GPU count: 1
GPU: NVIDIA A100-SXM4-80GB
```

A more complete GPU test can be saved as `gpu_test.py`:

```python
import os
import time
import torch

print("Node:", os.uname().nodename)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(i)

    print(
        f"GPU{i}: "
        f"{properties.name} "
        f"{properties.total_memory / 1024**3:.1f} GB"
    )

device = torch.device("cuda")

a = torch.randn(8192, 8192, device=device)
b = torch.randn(8192, 8192, device=device)

torch.cuda.synchronize()
start = time.time()

for _ in range(20):
    c = a @ b

torch.cuda.synchronize()

elapsed = time.time() - start

tflops = 20 * 2 * 8192**3 / elapsed / 1e12

print(
    f"matmul 8192^3 x20: "
    f"{elapsed:.2f}s -> "
    f"{tflops:.1f} TFLOPS"
)

os.makedirs("output", exist_ok=True)

torch.save(
    c.cpu()[:4, :4],
    "output/test_result.pt",
)

print("Output file written successfully.")
```

---

## 7. Cluster Limits and Storage

Current cluster configuration:

* GPU partition: `defq`
* One concurrently running job per user
* Up to three submitted jobs per user, including pending jobs
* User storage under `/home/<username>`
* Storage quota: 2 TB
* Tailscale authentication keys expire after 90 days
* `/tmp` is shared and has limited capacity

If a job remains in `PD` for an extended period, check the SLURM pending reason before contacting the cluster administrator.
