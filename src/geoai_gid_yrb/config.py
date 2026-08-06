from __future__ import annotations
from pathlib import Path
import os

def load_yaml(path):
    import yaml
    with open(path,'r',encoding='utf-8') as handle: return yaml.safe_load(handle)

def expand(value):
    return str(Path(os.path.expandvars(os.path.expanduser(str(value)))))
