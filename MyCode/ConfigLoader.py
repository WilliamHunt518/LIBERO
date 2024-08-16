import os
import yaml
from easydict import EasyDict
from hydra import compose, initialize
from omegaconf import OmegaConf

class ConfigLoader:
    def __init__(self, config_path="../libero/configs"):
        initialize(config_path=config_path)
        hydra_cfg = compose(config_name="MyConfig")
        yaml_config = OmegaConf.to_yaml(hydra_cfg)
        self.cfg = EasyDict(yaml.safe_load(yaml_config))
        self._setup_environment()

    def _setup_environment(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

    def get_config(self):
        return self.cfg
