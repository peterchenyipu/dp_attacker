from typing import Dict
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

class BaseImageRunner:
    def __init__(self, output_dir, attack_config=None):
        self.output_dir = output_dir
        self.attack_config = attack_config

    def run(self, policy: BaseImagePolicy) -> Dict:
        raise NotImplementedError()
