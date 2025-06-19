"""
DCASE 2025 full fine-tuned baseline model — Modular API for ASC inference.
"""

from typing import Optional, List
import torch
import torchaudio
import importlib.resources as pkg_resources
from torch import Tensor
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

# Model and resource imports
from Tan_SNTLNTU_task1.models.ntu_baseline_1 import get_ntu_model
from Tan_SNTLNTU_task1 import ckpts


class Config:
    """Configuration for audio preprocessing and model structure."""

    # Audio parameters
    orig_sample_rate = 44100
    sample_rate = 44100

    # Spectrogram parameters
    window_length = 8192
    hop_length = 1364
    n_fft = 8192
    n_mels = 256
    f_min = 0
    f_max = None

    # Model architecture
    n_classes = 10
    in_channels = 1
    base_channels = 24
    channels_multiplier = 1.5
    expansion_rate = 2
    divisor = 5

    # Device IDs
    device_ids = ['a', 'b', 'c', 's1', 's2', 's3']


class Baseline(torch.nn.Module):
    """
    DCASE 2025 Task 1 Baseline inference class for ASC with multi-device support.
    Includes log-mel preprocessing and model container setup.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        resample = torchaudio.transforms.Resample(
            orig_freq=self.config.orig_sample_rate,
            new_freq=self.config.sample_rate
        )

        # module to preprocess waveforms into log mel spectrograms
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.window_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max
        )

        self.mel = torch.nn.Sequential(
            resample,
            mel
        )   
        # Backbone model
        self.model = get_ntu_model(
            n_classes=config.n_classes,
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            channels_multiplier=config.channels_multiplier,
            expansion_rate=config.expansion_rate,
            divisor=config.divisor
        )

        # Multi-device wrapper: allows device-specific models
        #self.model = MultiDeviceModelContainer(base_model, devices=config.device_ids)
        #self.model = base_model
        self.model.eval()
        self.model.half()  # use float16 to meet complexity constraints

        self.class_order = [
            'airport', 'bus', 'metro', 'metro_station', 'park',
            'public_square', 'shopping_mall', 'street_pedestrian',
            'street_traffic', 'tram'
        ]

    def preprocess(self, waveform: Tensor) -> Tensor:
        """
        Convert raw waveform to log-mel spectrogram.

        Args:
            waveform: Tensor of shape [B, 1, n_samples]
        Returns:
            Tensor of shape [B, 1, n_mels, T]
        """
        mel = self.mel(waveform)
        return (mel + 1e-5).log().half()

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Perform forward pass for a batch of waveforms.

        Args:
            waveform: Tensor of shape [B, 1, n_samples]
            device_ids: List of device identifiers (length B)

        Returns:
            Tensor of shape [B, n_classes]
        """
        with torch.no_grad():
            mel = self.preprocess(waveform)  # [B, 1, n_mels, T]
            logits = self.model(mel)  # [B, n_classes]
        return logits


def load_model(model_file_path: Optional[str] = None) -> Baseline:
    """
    Load the full fine-tuned baseline model from a checkpoint.

    Args:
        model_file_path: Optional path to a .ckpt file. If None, uses the default packaged checkpoint.

    Returns:
        A Baseline model instance with loaded weights.
    """
    config = Config()
    model = Baseline(config)

    # Use default checkpoint from package resources if no path is given
    if model_file_path is None:
        with pkg_resources.path(ckpts, "task1_1.ckpt") as ckpt_path:
            model_file_path = str(ckpt_path)

    # Load checkpoint to CPU (compatible with CPU inference)
    ckpt = torch.load(model_file_path, map_location="cpu")

    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


def load_inputs(
    file_paths: List[str],
    device_ids: List[str],
    model: Baseline,
    num_workers: int = 1,
    batch_size: int = 256
) -> List[Tensor]:
    """
    Load and preprocess audio files in parallel with batch-wise resampling and STFT.

    Args:
        file_paths: List of .wav file paths.
        device_ids: List of corresponding device IDs (same length as file_paths).
        model: Baseline model (used for preprocessing).
        num_workers: Number of threads used for parallel loading.
        batch_size: Number of waveforms per batch for STFT/mel processing.

    Returns:
        List of mel spectrogram tensors [1, 1, n_mels, T], in same order as file_paths.
    """
    assert len(file_paths) == len(device_ids)

    device = next(model.parameters()).device
    target_sr = model.config.sample_rate

    def _load(indexed_path):
        path, idx = indexed_path
        waveform, sr = torchaudio.load(path)              # [channels, samples]
        waveform = waveform.mean(dim=0)                   # mono: [samples]
        return idx, waveform, sr

    # Step 1: Load & mono-mix in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        loaded = list(tqdm(
            exe.map(_load, zip(file_paths, range(len(file_paths)))),
            total=len(file_paths),
            desc="Loading files"
        ))
    # loaded = list of (original_idx, waveform, sample_rate)

    # Step 2: Group by original sample rate to batch resampling
    sr_groups = defaultdict(list)
    for idx, waveform, sr in loaded:
        sr_groups[sr].append((idx, waveform))

    inputs: List[Tensor] = [None] * len(file_paths)  # final output buffer

    # Step 3: Batch resample and preprocess each group
    print("Batched Resampling ...")
    for sr, items in sr_groups.items():
        print(f"Processing SR={sr} with {len(items)} files")
        for i in tqdm(range(0, len(items), batch_size), desc=f"SR={sr}", leave=False):
            chunk = items[i:i + batch_size]
            indices, waveforms = zip(*chunk)

            # Pad waveforms to the same length → [max_len, B]
            padded = pad_sequence(waveforms, batch_first=False)

            # Reshape → [B, 1, samples]
            batch_wave = padded.transpose(0, 1).unsqueeze(1).to(device)

            # Resample if needed
            if sr != target_sr:
                batch_wave = torchaudio.functional.resample(
                    batch_wave, orig_freq=sr, new_freq=target_sr
                )
            # STFT + mel computation for batch
            with torch.no_grad():
                mel_batch = model.preprocess(batch_wave)  # [B, 1, n_mels, T]

            # Store each preprocessed mel in original order
            for mel, idx in zip(mel_batch, indices):
                inputs[idx] = mel.unsqueeze(0).cpu()  # [1, 1, n_mels, T]

    return inputs


def get_model_for_device(
    model: Baseline,
    device_id: str
) -> torch.nn.Module:
    """
    Extract the device model corresponding to a specific device ID.

    Args:
        model: Baseline model instance.
        device_id: Device identifier string (e.g., 's1').

    Returns:
        The device model (nn.Module) associated with the given device.
    """
    device_model = model.model
    device_model.half()  # ensure float16 to meet complexity constraints
    return device_model


def predict(
    file_paths: List[str],
    device_ids: List[str],
    model_file_path: Optional[str] = None,
    use_cuda: bool = True,
    batch_size: int = 64
) -> List[torch.Tensor]:
    """
    Run inference on a list of audio files using device-specific models.

    Files are grouped by device ID and processed in batches.

    Args:
        file_paths: List of audio file paths.
        device_ids: List of device IDs corresponding to each file.
        model_file_path: Optional path to a model checkpoint (.ckpt).
        use_cuda: Whether to use GPU (if available).
        batch_size: Number of examples per inference batch.

Returns:
    A tuple (logits, class_order)
    - logits: List of tensors, one per file, each of shape [n_classes]
    - class_order: List of class names corresponding to the output logits
    """
    assert len(file_paths) == len(device_ids), "Number of files and device IDs must match."

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = load_model(model_file_path).to(device)

    # Step 1: Preprocess inputs → list of [1, 1, n_mels, T]
    inputs = load_inputs(file_paths, device_ids, model)

    # Step 2: Group by device ID, squeeze each mel to [n_mels, T]
    groups = defaultdict(list)
    for idx, (mel, dev) in enumerate(zip(inputs, device_ids)):
        mel_squeezed = mel.squeeze(0).squeeze(0)  # → [n_mels, T]
        groups[dev].append((mel_squeezed, idx))

    outputs = [None] * len(inputs)  # Placeholder for final predictions

    # Step 3: For each device, batch and infer
    for dev, items in tqdm(groups.items(), desc="Batched inference"):
        submodel = get_model_for_device(model, dev)

        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            mels, indices = zip(*chunk)  # List of [n_mels, T_i]

            # Pad to same length → [max_T, batch, n_mels]
            # This is to potentially also support files of varying length. However,
            # all files in the TAU dataset are exactly one second in length.
            padded = pad_sequence([m.T for m in mels], batch_first=False)

            # Reshape → [batch, 1, n_mels, max_T]
            batch = padded.permute(1, 2, 0).unsqueeze(1).to(device)

            with torch.no_grad():
                logits = submodel(batch).cpu()  # [B, n_classes]

            # Scatter outputs back in original file order
            for logit, idx in zip(logits, indices):
                outputs[idx] = logit

    return outputs, model.class_order
