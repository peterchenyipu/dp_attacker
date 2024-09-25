# for pre_generated_attacks, override the path to files by: 
#   python eval_generic.py --config-name=attack_config attack=pre_generated_attack attack.obs.agentview_image.path='aaaaa'

# for online attacks, override params by:
#   python eval_generic.py --config-name=attack_config attack=online_attack attack.alpha=0.001 attack.ntarget=null


# multi run:
# python eval_generic.py --config-name=attack_config --multirun '+experiment=default,transfer' target_ckpt_file_path=1,2 creates 4 runs!!!

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import torch
import dill
import numpy as np
from diffusion_policy.attack.adversarial_noise import AdversarialNoise
import os
import json
import wandb
import copy

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'attack_configs','eval_attack_config'))
)
def main(cfg: OmegaConf):
    print(OmegaConf.to_yaml(cfg))
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    # missing key check
    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if 'ckpt_path' in missing_keys:
        raise RuntimeError(f"missing key 'ckpt_path' in config file, pass it in as a command line argument 'ckpt_path=\"...\"'")

    OmegaConf.set_struct(cfg, False)
    # add and modify new fields here
    # cfg.test_new_field = 'test_new_field'
    OmegaConf.set_struct(cfg, True)

    print(OmegaConf.to_yaml(cfg))
    # return

    # load checkpoint
    payload = torch.load(open(cfg.ckpt.path, 'rb'), pickle_module=dill)
    workspace_cfg = payload['cfg']

    # complete patch path if physical attack
    if cfg.attack_type == 'physical_patch':
        cfg.attack.path = os.path.join(os.getcwd(), cfg.attack.path)

    # TODO
    # replace some components of the workspace_cfg
    workspace_cfg.logging = cfg.logging

    # make ddim scheduler for online attack if needed
    if cfg.attack.attack_type == 'online' and cfg.attack.online_pgd.type == 'full_chain' and cfg.attack.online_pgd.use_ddim:
        OmegaConf.set_struct(cfg, False)
        cfg.attack.online_pgd.scheduler = copy.deepcopy(workspace_cfg.policy.noise_scheduler)
        cfg.attack.online_pgd.scheduler['_target_'] = 'diffusers.schedulers.scheduling_ddim.DDIMScheduler'
        # remove the variance_type field
        cfg.attack.online_pgd.scheduler.pop('variance_type')
        OmegaConf.set_struct(cfg, True)
    elif cfg.attack.attack_type == 'online' and cfg.attack.online_pgd.type == 'full_chain' and not cfg.attack.online_pgd.use_ddim:
        OmegaConf.set_struct(cfg, False)
        cfg.attack.online_pgd.scheduler = None
        OmegaConf.set_struct(cfg, True)
    
    print(OmegaConf.to_yaml(cfg))
    # return
    cls = hydra.utils.get_class(workspace_cfg._target_)
    workspace = cls(workspace_cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if workspace_cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(cfg.device)
    policy.to(device)
    policy.eval()

    # lower the number of testing environments
    workspace_cfg.task.env_runner.n_train = 0
    workspace_cfg.task.env_runner.n_train_vis = 0

    # workspace_cfg.task.env_runner.n_test = 1
    # workspace_cfg.task.env_runner.n_test_vis = 1
    # workspace_cfg.task.env_runner.max_steps = 8
    # workspace_cfg.task.env_runner.n_envs = 1

    env_runner = hydra.utils.instantiate(
        workspace_cfg.task.env_runner,
        output_dir=workspace.output_dir,
        attack_config=cfg.attack if cfg.attack.attack_type is not None else None
    )
    # print(OmegaConf.to_yaml(env_runner.attack_config))
    # return

    noise = None
    simulated_physical_attack = None
    tf_dict = None
    # special case for pre-generated attacks
    if cfg.attack_type == 'pre_gen':
        # print(cfg.attack.pre_gen.obs.items())
        datadict = {key: np.load(value['path']) for key, value in cfg.attack.pre_gen.obs.items()}
        noise = AdversarialNoise(datadict=datadict)
        noise.to(device)
    
    # run the attack
    log = env_runner.run(policy, noise=noise, simulated_physical_attack=simulated_physical_attack, tf_dict=tf_dict)
    # log = env_runner.run_with_ntarget(policy, 1)
    # log = env_runner.run_stay_in_place(policy)


    if cfg.log_wandb:
        cfg = copy.deepcopy(cfg)
        OmegaConf.set_struct(cfg, False)
        cfg.ckpt_config = workspace_cfg # also log the original training config
        OmegaConf.set_struct(cfg, True)

        # configure logging
        wandb_run = wandb.init(
            dir=str(workspace.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": workspace.output_dir,
            }
        )

        # log the results
        wandb.log(log)
    
    # dump log to json
    json_log = dict()
    for key, value in log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value

    # append the attack config to the log
    
    json_log['cfg'] = OmegaConf.to_container(cfg)

    out_path = os.path.join(workspace.output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    


    # import time
    # time.sleep(30)
    print("done")

if __name__ == "__main__":
    main()
