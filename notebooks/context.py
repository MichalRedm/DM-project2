import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
)

from dm_project2 import dataset
from dm_project2 import preprocessing
from dm_project2 import clustering
