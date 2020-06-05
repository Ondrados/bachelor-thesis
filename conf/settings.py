import os
import sys
from decouple import config
from os.path import dirname, abspath

file_path = abspath(__file__)
root_path = dirname(dirname(file_path))

if root_path not in sys.path:
    sys.path.append(root_path)

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = config('BASE_DIR', default=root_path, cast=str)