import glob
import os
import sys

import json
from typing import Any

from src.models.taskers import Tasker

_RESULT_DIR = 'decode/vsr/vi'


class Reader(Tasker):

    def __init__(self):
        super().__init__()

    def do(self, sample: Any, *args, **kwargs) -> Any:
        decode_file = glob.glob(_RESULT_DIR + '/*.log')[0]

