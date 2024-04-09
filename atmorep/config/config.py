from pathlib import Path
from dataclasses import dataclass, asdict

_PATH_ROOT = Path(__file__).parent.parent.parent


YEAR_BASE = 1979
YEAR_LAST = 2022

_PATH_DATA = _PATH_ROOT / 'data'
_PATH_MODELS = _PATH_ROOT / 'models'
_PATH_RESULTS = _PATH_ROOT / 'results'

@dataclass
class PathConfig:
    root: Path
    data: Path
    models: Path
    results: Path
    
    def from_path_config(self, config: "PathConfig"):
        self.root = config.root
        self.data = config.data
        self.models = config.models
        self.results = config.results
        
    def __str__(self):
        return str(asdict(self))

PATHES = PathConfig(_PATH_ROOT, _PATH_DATA, _PATH_MODELS, _PATH_RESULTS)