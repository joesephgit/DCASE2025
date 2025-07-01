# DCASE2024 - Task 1 - Baseline Systems

Contact: **Ee Leng Tan** (etanel@ntu.edu.sg), *Nanyang Technological University*



## Abstract of Submission

We present the SNTL-NTU team’s submission for Task 1 of the Low-Complexity Acoustic Scene Classification (ASC) track in the Detection and Classification of Acoustic Scenes and Events (DCASE) 2025 Challenge. Departing from conventional teacher–student knowledge distillation (KD) strategies, our submission achieves competitive performance without employing KD, while strictly adhering to the model complexity constraints defined by the challenge.

The proposed system is a lightweight CNN-GRU architecture trained exclusively on the TAU Urban Acoustic Scenes 2022 Mobile Development dataset (25% split). No external datasets are used, except for MicIRP, which provides device impulse response (DIR) augmentation. The model has a memory footprint of just 117 KB and requires only 10.9 million multiply-and-accumulate (MAC) operations, making it one of the most efficient submissions to the challenge.

On the official development set, the model achieves an accuracy of 60.4% without device-specific finetuning. Training was conducted using the official DCASE 2024 Task 1 baseline codebase.


