import yaml


class Config:
    def __init__(self, config_dict):
        self.models = config_dict.get('models', {})
        self.search_config = config_dict.get('search_config', {})

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return cls(config_dict)
