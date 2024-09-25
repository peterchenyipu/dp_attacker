import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.attack.physical_attack import PatchedAdversarialNoise, PatchedAdversarialNoiseDict
from diffusion_policy.attack.CustomRandomAffine import CustomRandomAffine
import torchvision

class GetL2Workspace(BaseWorkspace):
    # include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, run_config: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)
        self.run_config = run_config
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        if run_config.ckpt.backbone_arch == 'CNN':
            self.optimizer = hydra.utils.instantiate(
                cfg.optimizer, params=self.model.parameters())
        else:
            self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        self.epoch = 0
        self.global_step = 0

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        print(f'{len(dataset)=}')

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        self.run_config = copy.deepcopy(self.run_config)
        OmegaConf.set_struct(self.run_config, False)
        self.run_config.ckpt_config = cfg # also log the original training config
        OmegaConf.set_struct(self.run_config, True)

        

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps_per_e = 3
            cfg.training.max_val_steps_per_e = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        vs_random_list = []
        vs_attacked_list = []
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(1):
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # print(f'{len(train_dataloader)=}')
                        # print(f'{batch_idx=}')
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        self.model.eval()
                        
                        raw_features = self.model.get_encoded_obs(batch['obs']).cpu() # B x C
                        # print(f'{raw_features.shape=}')
                        attacked_features = self.model.get_attacked_encoded_obs(batch['obs'], ntarget=1).cpu() # B x C
                        # print(f'{attacked_features.shape=}')
                        random_attacked_features = self.model.get_random_attacked_feature(batch['obs']).cpu() # B x C

                        # calculate the distance between the raw features and the attacked features
                        dist_raw_attacked = torch.linalg.vector_norm(raw_features - attacked_features, dim=1)
                        dist_raw_random_attacked = torch.linalg.vector_norm(raw_features - random_attacked_features, dim=1)
                        # print(f'{dist_raw_attacked.shape=}')
                        # print(f'{dist_raw_random_attacked.shape=}')
                        # print(f'{dist_raw_attacked.mean()=}')
                        # print(f'{dist_raw_random_attacked.mean()=}')
                        vs_attacked_list.append(dist_raw_attacked.detach().cpu())
                        vs_random_list.append(dist_raw_random_attacked.detach().cpu())

                        if batch_idx >= ((1000/20)-1):
                            break
                
                vs_attacked_dists = torch.cat(vs_attacked_list, dim=0)
                vs_random_dists = torch.cat(vs_random_list, dim=0)
                # save as npy
                np.save(os.path.join(self.output_dir, 'vs_attacked_dists.npy'), vs_attacked_dists.detach().numpy())
                np.save(os.path.join(self.output_dir, 'vs_random_dists.npy'), vs_random_dists.detach().numpy())


                
                self.global_step += 1
                self.epoch += 1

