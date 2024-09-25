import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.attack.adversarial_noise import AdversarialNoise
from diffusion_policy.attack.CustomRandomAffine import CustomRandomAffine
from diffusion_policy.attack.physical_attack import PatchedAdversarialNoise

import torchvision
from omegaconf import OmegaConf
from typing import Union

class PushTImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            attack_config=None,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            render_size=96,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir, attack_config=attack_config)
        if n_envs is None:
            n_envs = n_train + n_test

        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTImageEnv(
                        legacy=legacy_test,
                        render_size=render_size
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
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
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

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
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

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

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
        self.tqdm_interval_sec = tqdm_interval_sec
    
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
                                        or noise is not None or simulated_physical_attack is not None # fixed attack
        video_frames_dict = [None] * n_chunks

        if (self.attack_config is not None and self.attack_config.attack_type is not None):
            print("Attacking with config:")
            print(OmegaConf.to_yaml(self.attack_config))
            if (self.attack_config.online_pgd.type == 'full_chain'):
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

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtImageRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

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
                                    # B, T, Da=2
                                    ntarget = torch.ones(n_envs, policy.horizon, 2).to(device) * self.attack_config.online_pgd.ntarget
                                else:
                                    # throw exception if the target is not an int
                                    raise ValueError("only support ntarget as int for now")
                            else:
                                ntarget = None

                            if self.attack_config.online_pgd.type == 'random_step':
                                action_dict = policy.predict_action_attacked(obs_dict, eps=self.attack_config.eps, num_steps=self.attack_config.num_steps, alpha=self.attack_config.alpha, ntarget=ntarget)
                            elif self.attack_config.online_pgd.type == 'full_chain':
                                action_dict = policy.predict_action_full_chain_attacked(obs_dict, eps=self.attack_config.eps, 
                                                                                             num_steps=self.attack_config.num_steps, alpha=self.attack_config.alpha, 
                                                                                             ntarget=ntarget, scheduler=scheduler, 
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
                            # no attack
                            action_dict = policy.predict_action(obs_dict)
                    elif simulated_physical_attack is not None:
                        # add fixed noise
                        for key, value in tf_dict.items():
                            obs_dict[key] = simulated_physical_attack.stitch(obs_dict[key], tf_dict[key]['val_tf'])
                        action_dict = policy.predict_action(obs_dict)             
                    else:
                        # no attack
                        action_dict = policy.predict_action(obs_dict)

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
                        if 'attacked obs' in action_dict:
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

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

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

        if record_attacked_frames:
            for chunk_idx in range(n_chunks):
                # save video
                for key, video_frames_list in video_frames_dict[chunk_idx].items():
                    video_tensors = torch.stack(video_frames_list, dim=1)
                    for i in range(video_tensors.shape[0]):
                        video = video_tensors[i]
                        video_idx = i + chunk_idx * n_envs
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

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtImageRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
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

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

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
