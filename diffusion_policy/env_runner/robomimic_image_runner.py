import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import hydra
import math
import dill
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.attack.adversarial_noise import AdversarialNoise
from diffusion_policy.attack.CustomRandomAffine import CustomRandomAffine
from diffusion_policy.attack.physical_attack import PatchedAdversarialNoise
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import torchvision
from omegaconf import OmegaConf
from typing import Union
import pytorch3d.transforms as pt


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env


class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='sideview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            attack_config=None,
        ):
        super().__init__(output_dir, attack_config)

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        if attack_config is not None and attack_config.attack_type == 'physical_patch':
            # resize obs
            if env_meta['env_name'] != 'ToolHang':
                env_meta['env_kwargs']['camera_heights'] = 300
                env_meta['env_kwargs']['camera_widths'] = 300
                # from pprint import pprint
                # pprint(shape_meta)
                image_set = {}
                for key, value in shape_meta['obs'].items():
                    shape = value['shape']
                    if key.endswith('image'):
                        image_set[key] = [shape[0],  300, 300]
                for key, value in image_set.items():
                    shape_meta['obs'][key]['shape'] = value

            env_meta['env_kwargs']['patch_attack'] = True
            env_meta['env_kwargs']['patch_size'] = attack_config.patch_size
            env_meta['env_kwargs']['patch_path'] = attack_config.path



        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    shape_meta=shape_meta,
                    enable_render=False
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)


        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.shape_meta = shape_meta

    def run(self, policy: BaseImagePolicy, noise: PatchedAdversarialNoise=None, simulated_physical_attack: PatchedAdversarialNoise=None, tf_dict: dict[str, dict[str, CustomRandomAffine]]=None):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        elapsed_time_list = []
        
        # attacked video frames
        record_attacked_frames: bool = (self.attack_config is not None and self.attack_config.attack_type is not None) \
                                        or noise is not None or simulated_physical_attack is not None # fixed noise attack
        video_frames_dict = [None] * n_chunks

        if (self.attack_config is not None and self.attack_config.attack_type is not None):
            print("Attacking with config:")
            print(OmegaConf.to_yaml(self.attack_config))
            if (hasattr(self.attack_config, 'online_pgd') and self.attack_config.online_pgd.type == 'full_chain'):
                # instantiate the scheduler for the full chain attack
                scheduler = self.attack_config.online_pgd.scheduler
        else:
            print("No attack config provided, running regular policy rollout")

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                # obs keys: dict_keys(['sideview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'])
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                if self.attack_config is not None and self.attack_config.attack_type == 'physical_patch' and self.env_meta['env_name'] != 'ToolHang':
                    # resize all images
                    r_tf = torchvision.transforms.Resize(size=(84, 84))
                    value = next(iter(obs_dict.values()))
                    B, To = value.shape[:2]
                    for key in obs_dict.keys():
                        if 'image' in key:
                            # condition through global feature
                            obs_dict[key] = obs_dict[key][:,:To,...].reshape(-1,*obs_dict[key].shape[2:]) # flatten
                            obs_dict[key] = r_tf(obs_dict[key])
                            # reshape back to B, Do
                            obs_dict[key] = obs_dict[key].reshape(B, To, 3, 84, 84)


                # run policy
                with torch.no_grad():
                    if noise is not None:
                        # fixed noise attack
                        for key, noise_tensor in noise.noise_dict.items():
                            obs_dict[key] = obs_dict[key] + noise_tensor.unsqueeze(0)
                            obs_dict[key] = torch.clamp(obs_dict[key], 0, 1)
                        action_dict = policy.predict_action(obs_dict)

                    elif self.attack_config is not None:
                        if self.attack_config.attack_type == 'online':
                            # construct the target if needed
                            if self.attack_config.online_pgd.ntarget is not None:
                                if isinstance(self.attack_config.online_pgd.ntarget, int):
                                    # B, T, Da
                                    ntarget = torch.ones(n_envs, policy.horizon, *self.shape_meta.action.shape).to(device) * self.attack_config.online_pgd.ntarget
                                else:
                                    # throw exception if the target is not an int
                                    raise ValueError("only support ntarget as int for now")
                            else:
                                ntarget = None
                            if self.attack_config.online_pgd.type == 'random_step':
                                action_dict = policy.predict_action_attacked(obs_dict, eps=self.attack_config.eps, num_steps=self.attack_config.num_steps, alpha=self.attack_config.alpha, ntarget=ntarget)
                            elif self.attack_config.online_pgd.type == 'full_chain':
                                action_dict = policy.predict_action_full_chain_attacked(obs_dict, eps=self.attack_config.eps, num_steps=self.attack_config.num_steps,
                                                                                              alpha=self.attack_config.alpha, ntarget=ntarget, scheduler=scheduler,
                                                                                              scheduler_steps=self.attack_config.online_pgd.scheduler_steps)


                            elapsed_time_list.append(action_dict['attack_time'])
                            action_dict.pop('attack_time')
                        elif self.attack_config.attack_type == 'random_noise':
                            # random noise attack
                            for key, noise_tensor in obs_dict.items():
                                if key.endswith('image'):
                                    obs_dict[key] = obs_dict[key] + torch.clamp(torch.randn_like(obs_dict[key]) * self.attack_config.eps, -self.attack_config.eps, self.attack_config.eps)
                                    obs_dict[key] = torch.clamp(obs_dict[key], 0, 1)
                            
                            action_dict = policy.predict_action(obs_dict)
                        elif self.attack_config.attack_type == 'no_attack':
                            # no attack regular policy rollout
                            action_dict = policy.predict_action(obs_dict)
                        elif self.attack_config.attack_type == 'physical_patch':
                            action_dict = policy.predict_action(obs_dict) # patch already applied in env
                        else:
                            raise ValueError("Unknown attack type")
                    elif simulated_physical_attack is not None:
                        # add fixed noise
                        for key, value in tf_dict.items():
                            obs_dict[key] = simulated_physical_attack.stitch(obs_dict[key], tf_dict[key]['val_tf'])
                        action_dict = policy.predict_action(obs_dict)  
                    else:
                        # no attack regular policy rollout
                        action_dict = policy.predict_action(obs_dict)

                # action keys: dict_keys(['action', 'action_pred'])

                if record_attacked_frames:
                    # save attacked frames, convert to uint8
                    # first construct video frame list
                    if video_frames_dict[chunk_idx] is None:
                        video_frames_dict[chunk_idx] = dict()
                        # if self.attack_config is not None:
                        #     for key in action_dict['attacked obs'].keys():
                        #         if 'image' in key:
                        #             video_frames_dict[chunk_idx][key] = []
                        # else:
                        for key in obs_dict.keys():
                            if 'image' in key:
                                video_frames_dict[chunk_idx][key] = []
                    # convert to uint8
                    for key in video_frames_dict[chunk_idx].keys():
                        # dimension B, Do, C, H, W
                        if 'attacked obs' in action_dict.keys():
                            attacked_frames = (action_dict['attacked obs'][key] * 255).cpu().to(torch.uint8)
                        else:
                            attacked_frames = (obs_dict[key] * 255).cpu().to(torch.uint8)
                        # print(attacked_frames.shape)
                        V_B, V_Do, V_C, V_H, V_W = attacked_frames.shape
                        attacked_frames_permuted = attacked_frames.permute(0, 2, 3, 1, 4) # B, C, H, Do, W
                        attacked_frames_reshaped = attacked_frames_permuted.reshape(V_B, V_C, V_H, V_Do*V_W)
                        attacked_frames_reshaped = attacked_frames_reshaped.permute(0, 2, 3, 1) # B, H, Do*W, C
                        # print(attacked_frames_reshaped.shape)
                        video_frames_dict[chunk_idx][key].append(attacked_frames_reshaped)
                    
                    # # verify pixel difference
                    # print(torch.abs(action_dict['attacked obs']['agentview_image'] - obs_dict['agentview_image']).max())
                    # print(torch.abs(action_dict['attacked obs']['agentview_image'] - obs_dict['agentview_image']).mean())

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            
            
            
        # clear out video buffer
        _ = env.reset()

        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        if self.attack_config is not None:
            log_data['average elpased time per attack'] = np.mean(elapsed_time_list)
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # save attacked frames
        if record_attacked_frames:
            for chunk_idx in range(n_chunks):
                # save video
                for key, video_frames_list in video_frames_dict[chunk_idx].items():
                    video_tensors = torch.stack(video_frames_list, dim=1)
                    for i in range(video_tensors.shape[0]):
                        video = video_tensors[i]
                        video_idx = i + chunk_idx * n_envs
                        if (video_idx >= n_inits):
                            break
                        if all_video_paths[video_idx] is not None:
                            video_path = pathlib.Path(self.output_dir).joinpath(
                                'media', f'{pathlib.Path(all_video_paths[video_idx]).name}_attacked_{key}.mp4')
                            torchvision.io.write_video(str(video_path), video, fps=2)
                            attacked_video = wandb.Video(str(video_path))
                            seed = self.env_seeds[video_idx]
                            prefix = self.env_prefixs[video_idx]
                            log_data[prefix+f'sim_video_{seed}_attacked_{key}'] = attacked_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
    
    def run_with_ntarget(self, policy: BaseImagePolicy, ntarget: Union[int, np.ndarray]):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits


        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                # obs keys: dict_keys(['sideview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'])
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))


                # run policy
                value = next(iter(obs_dict.values()))
                B, To = value.shape[:2]
                Da = policy.action_dim

                n_action = np.ones((B, policy.n_action_steps, Da)) * ntarget
                action = policy.normalizer['action'].unnormalize(n_action).cpu().numpy()
                print(action.shape)
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)
                print(f'{env_action.shape=}')

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

           
            
        # clear out video buffer
        _ = env.reset()

        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def run_stay_in_place(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits


        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            

            np_obs_dict = dict(obs)
            if self.past_action and (past_action is not None):
                # TODO: not tested
                np_obs_dict['past_action'] = past_action[
                    :,-(self.n_obs_steps-1):].astype(np.float32)
            # obs keys: dict_keys(['sideview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'])
            # device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(
                    device=device))
            value = next(iter(obs_dict.values()))
            B, To = value.shape[:2]
            Da = policy.action_dim

            
            ee_positions = np_obs_dict['robot0_eef_pos'][:, 0, :] # B, 3
            ee_quats = np.zeros((B, 4)) # B, 4

            # set target as current position, note the convention difference, obs is in xyzw, pytorch3d expects wxyz
            ee_quats[:, 0] = np_obs_dict['robot0_eef_quat'][:, 0, -1] # B, 4
            ee_quats[:, 1:] = np_obs_dict['robot0_eef_quat'][:, 0, :-1] # B, 3

            # ee_quats[:, 0] = np.sqrt(2)/2
            # ee_quats[:, 2] = -np.sqrt(2)/2

            # somehow the action requires an offset rotation
            offset_quat = np.zeros((B, 4))
            offset_quat[:, 1] = -np.sqrt(2)/2
            offset_quat[:, 2] = -np.sqrt(2)/2

            ee_quats = pt.quaternion_multiply(torch.from_numpy(ee_quats), torch.from_numpy(offset_quat)).numpy()

            # ee_quats = pt.quaternion_invert(torch.from_numpy(ee_quats)).numpy()
            
            # ee_quats[:, 1] = -0.707
            # ee_quats[:, 2] = -0.707

            action = np.zeros((B, policy.n_action_steps, 7))
            tf_quat_to_aa = RotationTransformer('axis_angle', 'quaternion')
            # print(f'{action.shape=}')
            # return
            action[..., 3:3+3] = tf_quat_to_aa.inverse(ee_quats[:, None, :])
            action[..., :3] = ee_positions[:, None, :]

            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                # obs keys: dict_keys(['sideview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'])
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))


                # # run policy\
                ee_positions = np_obs_dict['robot0_eef_pos'][:, 0, :] # B, 3
                print(f'{ee_positions=}')
                ee_quats = np.zeros((B, 4)) # B, 4
                print(f'{np_obs_dict["robot0_eef_quat"][:, 0, :]=}')

                # # ee_quats[:,1:] = np_obs_dict['robot0_eef_quat'][:, 0, :3]
                # # ee_quats[:,0] = np_obs_dict['robot0_eef_quat'][:, 0, 3]
                
                # ee_quats[:, 1] = 1

                # print(f'{ee_quats=}')
                # # action = np.zeros((B, policy.n_action_steps, 7))
                # tf_quat_to_aa = RotationTransformer('quaternion', 'axis_angle')
                # # print(f'{action.shape=}')
                # # return
                # action[..., 3:3+3] = tf_quat_to_aa.forward(ee_quats[:, None, :])
                # action[..., :3] = ee_positions[:, None, :]
                
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                # if self.abs_action:
                #     env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

           
            
        # clear out video buffer
        _ = env.reset()

        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data


    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
