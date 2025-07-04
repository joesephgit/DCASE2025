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

The Baseline system (full fine-tune strategy) has a complexity of 58,605 parameters (16 bits) and 10.9 million MACs. The table below lists how the parameters
and MACs are distributed across the different layers in the network.

|==========================================================================================
|Layer (type:depth-idx)                   Output Shape              Param #
|==========================================================================================
|Network_test                             [1, 10]                   438|
|├─Conv2dNormActivation: 1-1              [1, 7, 256, 33]           --|
|│    └─Conv2d: 2-1                       [1, 7, 256, 33]           63|
|│    └─BatchNorm2d: 2-2                  [1, 7, 256, 33]           14
|├─Conv2dNormActivation: 1-2              [1, 12, 256, 33]          --
|│    └─Conv2d: 2-3                       [1, 12, 256, 33]          84
|│    └─BatchNorm2d: 2-4                  [1, 12, 256, 33]          24
|├─Conv2dNormActivation: 1-3              [1, 24, 256, 33]          --
|│    └─Conv2d: 2-5                       [1, 24, 256, 33]          72
|│    └─BatchNorm2d: 2-6                  [1, 24, 256, 33]          48
|├─Conv2dNormActivation: 1-4              [1, 24, 256, 33]          --
|│    └─Conv2d: 2-7                       [1, 24, 256, 33]          144
|│    └─BatchNorm2d: 2-8                  [1, 24, 256, 33]          48
|├─ChannelSELayer: 1-5                    [1, 24, 256, 33]          --
|│    └─Linear: 2-9                       [1, 12]                   300
|│    └─ReLU: 2-10                        [1, 12]                   --
|│    └─Linear: 2-11                      [1, 24]                   312
|│    └─Sigmoid: 2-12                     [1, 24]                   --
|├─MaxPool2d: 1-6                         [1, 24, 128, 16]          --
|├─AvgPool2d: 1-7                         [1, 24, 128, 16]          --
|├─Conv2dNormActivation: 1-8              [1, 36, 128, 16]          --
|│    └─Conv2d: 2-13                      [1, 36, 128, 16]          864
|│    └─BatchNorm2d: 2-14                 [1, 36, 128, 16]          72
|├─Conv2dNormActivation: 1-9              [1, 72, 128, 16]          --
|│    └─Conv2d: 2-15                      [1, 72, 128, 16]          360
|│    └─BatchNorm2d: 2-16                 [1, 72, 128, 16]          144
|├─Conv2dNormActivation: 1-10             [1, 72, 64, 8]            --
|│    └─Conv2d: 2-17                      [1, 72, 64, 8]            360
|│    └─BatchNorm2d: 2-18                 [1, 72, 64, 8]            144
|├─Conv2dNormActivation: 1-11             [1, 36, 128, 16]          --
|│    └─Conv2d: 2-19                      [1, 36, 128, 16]          864
|│    └─BatchNorm2d: 2-20                 [1, 36, 128, 16]          72
|├─Conv2dNormActivation: 1-12             [1, 72, 128, 16]          --
|│    └─Conv2d: 2-21                      [1, 72, 128, 16]          360
|│    └─BatchNorm2d: 2-22                 [1, 72, 128, 16]          144
|├─Conv2dNormActivation: 1-13             [1, 72, 64, 8]            --
|│    └─Conv2d: 2-23                      [1, 72, 64, 8]            360
|│    └─BatchNorm2d: 2-24                 [1, 72, 64, 8]            144
|├─ChannelSELayer: 1-14                   [1, 72, 64, 8]            --
|│    └─Linear: 2-25                      [1, 36]                   2,628
|│    └─ReLU: 2-26                        [1, 36]                   --
|│    └─Linear: 2-27                      [1, 72]                   2,664
|│    └─Sigmoid: 2-28                     [1, 72]                   --
|├─ChannelSELayer: 1-15                   [1, 72, 64, 8]            --
|│    └─Linear: 2-29                      [1, 36]                   2,628
|│    └─ReLU: 2-30                        [1, 36]                   --
|│    └─Linear: 2-31                      [1, 72]                   2,664
|│    └─Sigmoid: 2-32                     [1, 72]                   --
|├─MaxPool2d: 1-16                        [1, 72, 32, 4]            --
|├─AvgPool2d: 1-17                        [1, 72, 32, 4]            --
|├─AvgPool2d: 1-18                        [1, 72, 32, 4]            --
|├─MaxPool2d: 1-19                        [1, 72, 32, 4]            --
|├─Conv2dNormActivation: 1-20             [1, 72, 32, 4]            --
|│    └─Conv2d: 2-33                      [1, 72, 32, 4]            5,184
|│    └─BatchNorm2d: 2-34                 [1, 72, 32, 4]            144
|├─Conv2dNormActivation: 1-21             [1, 144, 16, 4]           --
|│    └─Conv2d: 2-35                      [1, 144, 16, 4]           720
|│    └─BatchNorm2d: 2-36                 [1, 144, 16, 4]           288
|├─Dropout2d: 1-22                        [1, 144, 16, 4]           --
|├─Conv2dNormActivation: 1-23             [1, 28, 16, 4]            --
|│    └─Conv2d: 2-37                      [1, 28, 16, 4]            20,160
|│    └─BatchNorm2d: 2-38                 [1, 28, 16, 4]            56
|├─GRU: 1-24                              [1, 28, 64]               15,744
|├─Dropout1d: 1-25                        [1, 28, 64]               --
|├─Conv1d: 1-26                           [1, 10, 1]                290
||==========================================================================================
|Total params: 58,605
