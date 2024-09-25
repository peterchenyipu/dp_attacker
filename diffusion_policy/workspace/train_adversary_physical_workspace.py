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

class TrainPhysicalAdversarialNoiseWorkspace(BaseWorkspace):
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

        tf_dict = {}
        for key, value in self.run_config.attack_config.attacked_obs.items(): 
            tf_dict[key] = {}
            tf_dict[key]['train_tf'] = hydra.utils.instantiate(value.train_tf)
            tf_dict[key]['val_tf'] = hydra.utils.instantiate(value.val_tf)
        
        patch = PatchedAdversarialNoise(dim=tuple(self.run_config.attack_config.shape))
        if (self.run_config.attack_config.random_start):
            patch.noise = torch.rand_like(patch.noise)

        self.run_config = copy.deepcopy(self.run_config)
        OmegaConf.set_struct(self.run_config, False)
        self.run_config.ckpt_config = cfg # also log the original training config
        OmegaConf.set_struct(self.run_config, True)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(self.run_config, resolve=True),
            **self.run_config.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

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

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        self.model.eval()
                        
                        # perturb the observation
                        patch = patch.copy()
                        # to device
                        patch.to(device)
                        patch.enable_grad()
                        for key, value in tf_dict.items():
                            batch['obs'][key] = patch.stitch(batch['obs'][key], tf_dict[key]['train_tf'])
                        
                        # method 1: random step loss
                        if self.run_config.attack_config.ntarget is not None:
                            if isinstance(self.run_config.attack_config.ntarget, int):
                                raw_loss = self.model.compute_targeted_loss(batch, self.run_config.attack_config.ntarget)
                            else:
                                # throw exception if the target is not an int
                                raise ValueError("only support ntarget as int for now")
                        else:
                            # non-targeted attack
                            raw_loss = self.model.compute_loss(batch)
                            raw_loss = - raw_loss

                        raw_loss.backward()

                        if self.run_config.attack_method.name == 'pgd':
                            # apply pgd on the noise
                            new_gradient = patch.noise.grad.data.sign()
                            # print(f'{new_gradient.mean()=}')
                            patch.noise = patch.noise - self.run_config.attack_method.alpha * new_gradient
                        elif self.run_config.attack_method.name == 'gd':
                            # apply sgd on the noise
                            new_gradient = patch.noise.grad.data
                            # print(f'{new_gradient.mean()=}')
                            patch.noise = patch.noise - self.run_config.attack_method.lr * new_gradient
                        # noise.noise = noise.noise - 0.1 * new_gradient
                        patch.noise = torch.clamp(patch.noise, 0, 1)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                        }


                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps_per_e is not None) \
                            and batch_idx >= (cfg.training.max_train_steps_per_e-1):
                            print(OmegaConf.to_container(cfg, resolve=True))
                            print("Early stopping training")
                            break

                        # checkpoint
                        if (batch_idx % cfg.training.checkpoint_every) == 0:
                            # save the noise
                            noise_cpy = patch.copy()
                            
                            np.save(os.path.join(self.output_dir,
                                                    f'e={local_epoch_idx}_b={batch_idx}.npy'), 
                                                    noise_cpy.noise.cpu().detach().numpy())
                            torchvision.io.write_png((noise_cpy.noise.cpu().detach() * 255).to(torch.uint8), 
                                                    os.path.join(self.output_dir,
                                                    f'e={local_epoch_idx}_b={batch_idx}.png'))
                            step_log['patch image'] = wandb.Image(os.path.join(self.output_dir,
                                                    f'e={local_epoch_idx}_b={batch_idx}.png'))
                        elif is_last_batch:
                            # save the noise
                            noise_cpy = patch.copy()
                            np.save(os.path.join(self.output_dir,
                                                    f'e={local_epoch_idx}_b={batch_idx}.npy'), 
                                                    noise_cpy.noise.cpu().detach().numpy())
                            torchvision.io.write_png((noise_cpy.noise.cpu().detach() * 255).to(torch.uint8), 
                                                    os.path.join(self.output_dir,
                                                    f'e={local_epoch_idx}_b={batch_idx}.png'))
                            step_log['patch image'] = wandb.Image(os.path.join(self.output_dir,
                                                    f'e={local_epoch_idx}_b={batch_idx}.png'))
                            # sanitize metric names
                            metric_dict = dict()
                            for key, value in step_log.items():
                                new_key = key.replace('/', '_')
                                metric_dict[new_key] = value
                
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss


                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy, simulated_physical_attack=patch, tf_dict=tf_dict)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                                for key, value in tf_dict.items():
                                    batch['obs'][key] = patch.stitch(batch['obs'][key], tf_dict[key]['val_tf'])
                                
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps_per_e is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps_per_e-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        # attack
                        for key, value in tf_dict.items():
                            batch['obs'][key] = patch.stitch(batch['obs'][key], tf_dict[key]['val_tf'])
                                        
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

