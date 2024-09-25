import torch
import torch.nn as nn
import numpy as np
from typing import Union
from diffusion_policy.attack.CustomRandomAffine import CustomRandomAffine
from typing import Optional, Union

class PatchedAdversarialNoise():
    def __init__(self, dim: tuple[int], tf: CustomRandomAffine=None) -> None:
        """
        
        Args:
            dim (tuple[int]): The dimensions of the noise tensor. Should be 3, height, width
        """
        self.noise = torch.zeros(dim)
        if tf is not None:
            self.transform = tf
        else:
            self.transform = CustomRandomAffine(0)

    def enable_grad(self):
        """Enables gradient computation for the noise tensor."""
        self.noise.requires_grad = True

    def disable_grad(self):
        """Disables gradient computation for the noise tensor."""
        self.noise.requires_grad = False

    def copy(self):
        """Returns a deep copy of the AdversarialNoise object."""
        new_noise = PatchedAdversarialNoise(self.noise.shape)
        new_noise.noise = self.noise.clone().detach()
        return new_noise
    
    def to(self, device: torch.device):
        """Moves the noise tensor to the specified device.
        
        Args:
            device (torch.device): The device to move the noise tensor to.
        """
        self.noise = self.noise.to(device)

    def stitch(self, image: torch.Tensor, tf: CustomRandomAffine=None):
        """Stitches the noise to the image.
        
        Args:
            image (torch.Tensor): The image tensor.
            tf (CustomRandomAffine): The transformation object.
        """
        if tf is not None:
            self.transform = tf
        image = image.clone()
        
        # expand if needed
        if (len(image.shape) == 3):
            image = image.unsqueeze(0)
        B = image.shape[0]
        
        image_h, image_w = image.shape[-2:]
        noise_h, noise_w = self.noise.shape[-2:]
        condense = False
        
        if len(image.shape) == 5:
            #B, T, C, H, W
            condense = True
            image = image.reshape(-1, *image.shape[-3:])
        
        empty = torch.zeros((image.shape[0], image.shape[-3] + 1, image_h, image_w)).to(image)
        empty[:, :3, (n_h:=image_h//2 - noise_h//2):n_h+noise_h, (n_w:=image_w//2 - noise_w//2):n_w+noise_w] = self.noise
        empty[:, -1, (n_h:=image_h//2 - noise_h//2):n_h+noise_h, (n_w:=image_w//2 - noise_w//2):n_w+noise_w] = 1
        centered_noise = empty

        affined_noise = self.transform(centered_noise)
        mask = affined_noise[:, -1, ...] != 0
        mask = mask.unsqueeze(1).expand(-1, 3, -1, -1)
        image[mask] = affined_noise[:, :3, ...][mask]
        image = image.clamp(0, 1)

        if condense:
            image = image.reshape(B, -1, *image.shape[-3:])

        return image


class PatchedAdversarialNoiseDict():
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
                self.noise_dict[key] = PatchedAdversarialNoise(dim=shape)

        if datadict is not None:
            self.noise_dict = {}
            for key, data in datadict.items():
                if isinstance(data, torch.Tensor):
                    self.noise_dict[key] = PatchedAdversarialNoise(dim=data.shape)
                    self.noise_dict[key].noise = data.clone()
                elif isinstance(data, np.ndarray):
                    self.noise_dict[key] = PatchedAdversarialNoise(dim=data.shape)
                    self.noise_dict[key].noise = torch.from_numpy(data).clone()
    
    def enable_grad(self):
        """Enables gradient computation for the noise tensors."""
        for key in self.noise_dict.keys():
            self.noise_dict[key].enable_grad()
    
    def disable_grad(self):
        """Disables gradient computation for the noise tensors."""
        for key in self.noise_dict.keys():
            self.noise_dict[key].disable_grad()
    
    def copy(self):
        """Returns a deep copy of the AdversarialNoise object."""
        new_noise = PatchedAdversarialNoiseDict(self.shape_meta)
        for key in self.noise_dict.keys():
            new_noise.noise_dict[key] = self.noise_dict[key].copy()
        return new_noise
    
    def to(self, device: torch.device):
        """Moves the noise tensors to the specified device.
        
        Args:
            device (torch.device): The device to move the noise tensors to.
        """
        for key in self.noise_dict.keys():
            self.noise_dict[key].to(device)

    
    def __getitem__(self, key: str) -> PatchedAdversarialNoise:
        return self.noise_dict[key]
    
    def __setitem__(self, key: str, value: PatchedAdversarialNoise):
        self.noise_dict[key] = value
            
            

        