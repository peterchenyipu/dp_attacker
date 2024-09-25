"""
Usage:
Training:
python train_attack_img.py --config-name=attack
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_adversary_physical_workspace import TrainPhysicalAdversarialNoiseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'attack_configs','train_physical_attack_config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if 'ckpt_path' in missing_keys:
        raise RuntimeError(f"missing key 'ckpt_path' in config file, pass it in as a command line argument 'ckpt_path=\"...\"'")


    OmegaConf.set_struct(cfg, False)
    # add and modify new fields here
    cfg.test_new_field = 'test_new_field'



    OmegaConf.set_struct(cfg, True)


    print(OmegaConf.to_yaml(cfg))

    # load checkpoint
    payload = torch.load(open(cfg.ckpt.path, 'rb'), pickle_module=dill)
    workspace_cfg = payload['cfg']

    # replace some components of the workspace_cfg
    workspace_cfg.dataloader = cfg.dataloader
    workspace_cfg.val_dataloader = cfg.val_dataloader
    workspace_cfg.logging = cfg.logging

    OmegaConf.set_struct(workspace_cfg, False)
    for key, value in cfg.training.items():
        workspace_cfg.training[key] = value
    OmegaConf.set_struct(workspace_cfg, True)
    # workspace_cfg.training = cfg.training
    print(OmegaConf.to_yaml(workspace_cfg))

    # # lower the number of testing environments
    workspace_cfg.task.env_runner.n_train = 0
    workspace_cfg.task.env_runner.n_train_vis = 0

    # workspace_cfg.task.env_runner.n_test = 1
    # workspace_cfg.task.env_runner.n_test_vis = 1
    # workspace_cfg.task.env_runner.max_steps = 20
    # workspace_cfg.task.env_runner.n_envs = 28


    workspace: TrainPhysicalAdversarialNoiseWorkspace = TrainPhysicalAdversarialNoiseWorkspace(workspace_cfg, run_config=cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    workspace.run()

if __name__ == "__main__":
    main()
