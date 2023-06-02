import os
from typing import Tuple, Any

import pandas as pd
from procthor.databases import (
    DatabaseLoader,
    ProcTHORDatabase,
    keydefaultdict,
)

from procthor.objaverse.create_databases_with_objaverse import OBJAVERSE_DIR

MAX_HEAD_OBJAVERSE_OBJECT_TYPES_PER_ROOM = 4  # 20
MAX_TAIL_OBJAVERSE_OBJECT_TYPES_PER_ROOM = 1  # 20
MIN_OBJAVERSE_INSTANCES_FOR_HEAD_CATEGORY = 3
EXCLUDE_NON_OBJAVERSE_ASSETS = False

ALLOW_DUPLICATE_OBJAVERSE_WALL_OBJECTS_IN_HOUSE = False
OBJAVERSE_WALL_OBJECTS_PER_ROOM = {
    "population": [0, 1, 2, 3, 4],
    "weights": [0.50, 0.50, 0.00, 0.00, 0.00],
}

OBJAVERSE_DATASETS_DIR = os.environ.get(
    "OBJAVERSE_DATASETS_DIR", os.path.join(OBJAVERSE_DIR, "objaverse_databases")
)


def _objaverse_get_floor_assets(
    room_type: str, split: str, pt_db: ProcTHORDatabase
) -> Tuple[Any, pd.DataFrame]:
    floor_types = pt_db.PLACEMENT_ANNOTATIONS[
        pt_db.PLACEMENT_ANNOTATIONS["onFloor"]
        & (pt_db.PLACEMENT_ANNOTATIONS[f"in{room_type}s"] > 0)
    ]
    assets = pd.DataFrame(
        [
            {
                "assetId": asset["assetId"],
                "assetType": asset["objectType"],
                "split": asset["split"],
                "xSize": asset["boundingBox"]["x"],
                "ySize": asset["boundingBox"]["y"],
                "zSize": asset["boundingBox"]["z"],
            }
            for asset_type in floor_types.index
            for asset in pt_db.ASSET_DATABASE[asset_type]
        ]
    )
    assets = pd.merge(assets, floor_types, on="assetType", how="left")

    if split == "train":
        assets = assets[assets["split"].isin([split, None])]
    else:
        assets = assets[assets["split"] == split]

    assets.set_index("assetId", inplace=True)

    return floor_types, assets


def _get_objaverse_floor_assets_from_key(key: Tuple[str, str]):
    return _objaverse_get_floor_assets(*key, pt_db=DEFAULT_OBJAVERSE_PROCTHOR_DATABASE)


_DDL = DatabaseLoader(databases_dir=OBJAVERSE_DATASETS_DIR)

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
    FLOOR_ASSET_DICT=keydefaultdict(_get_objaverse_floor_assets_from_key),
    PRIORITY_ASSET_TYPES={
        "Bedroom": ["Bed", "Dresser"],
        "LivingRoom": ["Television", "DiningTable", "Sofa"],
        "Kitchen": ["CounterTop", "Fridge"],
        "Bathroom": ["Toilet", "Sink"],
    },
)
