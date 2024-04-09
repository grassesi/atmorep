from atmorep.core.evaluator import Evaluator
from atmorep.config.config import PATHES, PathConfig


class Atmorep:
    def __init__(self, pathes: PathConfig):
        PATHES.from_path_config(pathes)
        self.evaluator = Evaluator