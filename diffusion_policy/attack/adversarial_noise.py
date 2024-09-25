import torch
import torch.nn as nn
import numpy as np
from typing import Union

class AdversarialNoise():
    def __init__(self, shape_meta: dict[str, tuple[int,...]]=None, datadict: dict[str, Union[torch.Tensor, np.ndarray]]=None) -> None:
        """
        
        Args:
            shape_meta (dict[str, tuple[int,...]]): Dictionary of shape metadata for the noise tensor.
            datadict (dict[str, torch.Tensor|np.ndarray]): Dictionary of data tensors to be used for initialization.
        """
        if shape_meta is None and datadict is None:
            raise ValueError("Either shape_meta or datadict must be provided.")
        
        if shape_meta is not None:
            self.shape_meta = shape_meta
            self.noise_dict = {}
            for key, shape in shape_meta.items():
                self.noise_dict[key] = torch.zeros(shape)
        
        if datadict is not None:
            self.noise_dict = {}
            for key, data in datadict.items():
                if isinstance(data, torch.Tensor):
                    self.noise_dict[key] = data.clone()
                elif isinstance(data, np.ndarray):
                    self.noise_dict[key] = torch.from_numpy(data).clone()
                
    
    def enable_grad(self):
        """Enables gradient computation for the noise tensors."""
        for key in self.noise_dict.keys():
            self.noise_dict[key].requires_grad = True
    
    def disable_grad(self):
        """Disables gradient computation for the noise tensors."""
        for key in self.noise_dict.keys():
            self.noise_dict[key].requires_grad = False
    
    def copy(self):
        """Returns a deep copy of the AdversarialNoise object."""
        new_noise = AdversarialNoise(self.shape_meta)
        for key in self.noise_dict.keys():
            new_noise.noise_dict[key] = self.noise_dict[key].clone().detach()
        return new_noise
    
    def to(self, device: torch.device):
        """Moves the noise tensors to the specified device.
        
        Args:
            device (torch.device): The device to move the noise tensors to.
        """
        for key in self.noise_dict.keys():
            self.noise_dict[key] = self.noise_dict[key].to(device)
    
    def __getitem__(self, key: str) -> torch.Tensor:
        return self.noise_dict[key]
    
    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self.noise_dict[key] = value

    