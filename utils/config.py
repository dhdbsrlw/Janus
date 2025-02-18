import os
import sys # add
from datetime import datetime
from omegaconf import OmegaConf
from utils import AttrDict

def override_from_file_name(cfg):
    if 'cfg_path' in cfg and cfg.cfg_path is not None:
        file_cfg = OmegaConf.load(cfg.cfg_path)
        cfg = OmegaConf.merge(cfg, file_cfg)
    return cfg

def override_from_cli(cfg):
    c = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, c)
    return cfg

def to_attr_dict(cfg):
    c = OmegaConf.to_container(cfg)
    cfg = AttrDict.from_nested_dicts(c)
    return cfg

def build_config(struct=False, cfg_path="configs/train.yaml"):
    if cfg_path is None:
        raise ValueError("No cfg_path given.")
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, struct)
    cfg = override_from_file_name(cfg)
    cfg = override_from_cli(cfg)

    cfg_yaml = OmegaConf.to_yaml(cfg)
    cfg = to_attr_dict(cfg) # TODO: using attr class in OmegaConf?

    # return cfg, cfg_yaml
    return cfg
