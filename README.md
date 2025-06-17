# DCASE2024 - Task 1 - Baseline Systems

Contact: **Ee Leng Tan** (etanel@ntu.edu.sg), *Nanyang Technological Unveristy*



## Abstract of Submission

we present the SNTL-NTU team’s Task 1 submission for the Low-Complexity Acoustic Scene Classification of the Detection and Classification of Acoustic Scenes and Events (DCASE) 2025 challenge [1]. This submission departs from the typical application of knowledge distillation from a teacher to a student model, aiming to achieve high performance with limited complexity. The proposed model is based on a CNN-GRU model and is trained solely using the TAU Urban Acoustic Scene 2022 Mobile development dataset [2], without utilizing any external datasets, except for MicIRP [3], which is used for device impulse response (DIR) augmentation. The proposed model has a memory usage of 114.2 KB and requires 10.9M multiply-and-accumulate (MAC) operations. Using the development dataset, the proposed model achieved an accuracy of 60.25%



