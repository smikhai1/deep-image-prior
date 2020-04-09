import os
import yaml

class Config:

    def __init__(self, config_path):

        if not os.path.exists(config_path):
            raise FileNotFoundError('Config file wasn\'t found at {}'.format(config_path))
        if not os.path.isfile(config_path):
            raise ValueError('Not a file {}'.format(config_path))

        with open(config_path, 'r') as handle:
            config = yaml.load(handle, Loader=yaml.BaseLoader)

        for key in config:
            setattr(self, key, config[key])