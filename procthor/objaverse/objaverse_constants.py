import os
from pathlib import Path

MAX_HEAD_OBJAVERSE_OBJECT_TYPES_PER_ROOM = 4  # 20
MAX_TAIL_OBJAVERSE_OBJECT_TYPES_PER_ROOM = 1  # 20
MIN_OBJAVERSE_INSTANCES_FOR_HEAD_CATEGORY = 3
EXCLUDE_NON_OBJAVERSE_ASSETS = False

ALLOW_DUPLICATE_OBJAVERSE_WALL_OBJECTS_IN_HOUSE = False
OBJAVERSE_WALL_OBJECTS_PER_ROOM = {
    "population": [0, 1, 2, 3, 4],
    "weights": [0.50, 0.50, 0.00, 0.00, 0.00],
}

OBJAVERSE_DIR = os.path.abspath(os.path.dirname(Path(__file__)))
OBJAVERSE_DATASETS_DIR = os.environ.get(
    "OBJAVERSE_DATASETS_DIR", os.path.join(OBJAVERSE_DIR, "objaverse_databases")
)
