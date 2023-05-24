import os
from typing import Tuple

from procthor.databases import (
    _get_floor_assets,
    DatabaseLoader,
    ProcTHORDatabase,
    keydefaultdict,
)

MAX_HEAD_OBJAVERSE_OBJECT_TYPES_PER_ROOM = 4 #20
MAX_TAIL_OBJAVERSE_OBJECT_TYPES_PER_ROOM = 1 #20
MIN_OBJAVERSE_INSTANCES_FOR_HEAD_CATEGORY = 3
EXCLUDE_NON_OBJAVERSE_ASSETS = False

ALLOW_DUPLICATE_OBJAVERSE_WALL_OBJECTS_IN_HOUSE = False
OBJAVERSE_WALL_OBJECTS_PER_ROOM = {
    "population": [0, 1, 2, 3, 4],
    "weights": [0.50, 0.50, 0.00, 0.00, 0.00],
}

def _get_default_floor_assets_from_key(key: Tuple[str, str]):
    return _get_floor_assets(*key, pt_db=DEFAULT_OBJAVERSE_PROCTHOR_DATABASE)



_DDL = DatabaseLoader(databases_dir=os.path.join(os.path.dirname(__file__), "objaverse_databases"))

DEFAULT_OBJAVERSE_PROCTHOR_DATABASE = ProcTHORDatabase(
    SOLID_WALL_COLORS=_DDL.get_solid_wall_colors(),
    MATERIAL_DATABASE=_DDL.get_material_database(),
    SKYBOXES=_DDL.get_skyboxes(),
    OBJECTS_IN_RECEPTACLES=_DDL.get_object_in_receptacles(),
    ASSET_DATABASE=_DDL.get_asset_database(),
    ASSET_ID_DATABASE=_DDL.get_asset_id_database(),
    PLACEMENT_ANNOTATIONS=_DDL.get_placement_annotations(),
    # AI2THOR_OBJECT_METADATA=_DDL._get_ai2thor_object_metadata(),
    ASSET_GROUPS=_DDL.get_asset_groups(),
    ASSETS_DF=_DDL.get_assets_df(),
    WALL_HOLES=_DDL.get_wall_holes(),
    FLOOR_ASSET_DICT=keydefaultdict(_get_default_floor_assets_from_key),
    PRIORITY_ASSET_TYPES={
        "Bedroom": ["Bed", "Dresser"],
        "LivingRoom": ["Television", "DiningTable", "Sofa"],
        "Kitchen": ["CounterTop", "Fridge"],
        "Bathroom": ["Toilet", "Sink"],
    },
)
