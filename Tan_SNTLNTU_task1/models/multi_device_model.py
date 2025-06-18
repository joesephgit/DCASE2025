import torch
import torch.nn as nn
import copy
from typing import Tuple


class MultiDeviceModelContainer(nn.Module):
    """
    Multiple device fine-tuning container.
    """
    def __init__(self, base_model: nn.Module, devices: list):
        """
        Initializes the container with a base model.

        Args:
            base_model (nn.Module): The base model to be adapted per device.
            devices (list): A list of device identifiers (e.g., ["a", "b", "c"]).
        """
        super().__init__()
        self.base_model = base_model
        self.devices = devices

        # Create device-specific models
        self.device_models = nn.ModuleDict({
            device: copy.deepcopy(base_model) for device in devices
        })

    def forward(self, x: torch.Tensor, devices: Tuple[str] = None) -> torch.Tensor:
        """
        Forward pass through the model specific to the given device.

        Args:
            x (torch.Tensor): Input tensor.
            devices (Tuple[str]): Tuple of device identifiers corresponding to each sample.

        Returns:
            torch.Tensor: The model output.
        """
        if devices is None:
            # No device info → use base model
            return self.base_model(x)
        elif len(set(devices)) > 1:
            # More than one device in batch → forward one sample at a time
            return self._forward_multi_device(x, devices)
        elif devices[0] in self.device_models:
            # Single known device → use device-specific model
            return self.get_model_for_device(devices[0])(x)
        else:
            # Single unknown device → fall back to base model
            return self.base_model(x)

    def _forward_multi_device(self, x: torch.Tensor, devices: Tuple[str]) -> torch.Tensor:
        """
        Handles forward pass when multiple devices are present in the batch.
        """
        outputs = [self.device_models[device](x[i].unsqueeze(0)) if device in self.device_models
                   else self.base_model(x[i].unsqueeze(0))
                   for i, device in enumerate(devices)]
        return torch.cat(outputs)

    def get_model_for_device(self, device_name: str) -> nn.Module:
        """
        Retrieve the model corresponding to a specific device.

        Args:
            device_name (str): The device identifier.

        Returns:
            nn.Module: The model corresponding to the device.
        """
        if device_name in self.device_models:
            return self.device_models[device_name]
        else:
            return self.base_model
