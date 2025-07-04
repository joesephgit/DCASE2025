# DCASE2025 - Task 1 - SNTL_NTU_task1

Contact: **Ee-Leng TAN** (etanel@ntu.edu.sg), *Nanyang Technological University *

## Task 1 Submission of SNTL_NTU

We present the SNTL-NTU team’s submission for Task 1 of the Low-Complexity Acoustic Scene Classification (ASC) track in the Detection and Classification of Acoustic Scenes and Events (DCASE) 2025 Challenge. Departing from conventional teacher–student knowledge distillation (KD) strategies, our submission achieves competitive performance without employing KD, while strictly adhering to the model complexity constraints defined by the challenge.

The proposed system is a lightweight CNN-GRU architecture trained exclusively on the TAU Urban Acoustic Scenes 2022 Mobile Development dataset (25% split). No external datasets are used, except for MicIRP, which provides device impulse response (DIR) augmentation. The model has a memory footprint of just 117 KB and requires only 10.9 million multiply-and-accumulate (MAC) operations, making it one of the most efficient submissions to the challenge.

On the official development set, the model achieves an accuracy of 60.35% without device-specific finetuning, and all weights are shared across all 7 models during inference. Training was conducted using a modified DCASE 2025 Task 1 baseline codebase (train_base_dcase_2025.py).
 

### **Training Steps Overview**
#### **Step 1: General Model Training ([`train_base_dcase_2025.py`](train_base_dcase_2025.py))**
- Trains a **single baseline model** using the **25% subset** of the dataset.
- Focuses on **cross-device generalization**.
- No device-specific adaptation is performed.

#### **Step 2: Device-Specific Fine-Tuning ([`train_device_specific.py`](train_device_specific.py))**
This step **loads the pre-trained baseline model** from Step 1 and **fine-tunes it separately for each recording device** on the device-specific 
data contained in the 25% split.  
The approach consists of the following steps:
1. **Load the pre-trained checkpoint** from Step 1.
2. **Load device-specific training and test sets**.
3. **Iterate over all training devices** and train a specialized model for each (fine-tune all model parameters).
4. Compute overall peformance using the device-specific models.
5. **Handle unseen devices**: The **base model from Step 1** is used for devices not in the training set.

This two-stage approach ensures that the model **learns a general representation first**, before **adapting to specific device characteristics**.

## Getting Started

1. Clone this repository.
2. Create and activate a [conda](https://docs.anaconda.com/free/miniconda/index.html) environment:

```
conda create -n d25_t1 python=3.11
conda activate d25_t1
```

3. Install [PyTorch](https://pytorch.org/get-started/previous-versions/) version that suits your system. For example:

```
# for example:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or for the most recent versions:
pip3 install torch torchvision torchaudio
```

4. Install requirements:

```
pip3 install -r requirements.txt
```

5. Download and extract the [TAU Urban Acoustic Scenes 2022 Mobile, Development dataset](https://zenodo.org/records/6337421).

You should end up with a directory that contains, among other files, the following:
* A directory *audio* containing 230,350 audio files in *wav* format
* A file *meta.csv* that contains 230,350 rows with columns *filename*, *scene label*, *identifier* and *source label*

6. Specify the location of the dataset directory in the variable *dataset_dir* in file [dataset/dcase25.py](dataset/dcase25.py).
7. If you have not used [Weights and Biases](https://wandb.ai/site) for logging before, you can create a free account. On your
machine, run ```wandb login``` and copy your API key from [this](https://wandb.ai/authorize) link to the command line.

## Training Process

After training a **general model** in Step 1, you can fine-tune it for specific devices (Step 2).  

### Full Device-specific Fine-Tuning

**Step 1:** Train a **general model** on the full training set to maximize cross-device generalization.

```
python train_base_dcase_2025.py
```

**Step 2:** Load the pre-trained model from **Step 1** and fine-tune for all devices in the training set (`a`, `b`, `c`, `s1`, `s2`, `s3`):

```
python train_device_specific.py --ckpt_id=<wandb_id_from_Step_1>
```

## Baseline Complexity

The Baseline system (full fine-tune strategy) has a complexity of 58,605 parameters (16 bits) and 10.9 million MACs.
