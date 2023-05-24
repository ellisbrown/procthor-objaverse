import json
import os
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Union, Tuple, Optional

import numpy as np
import pandas as pd
from ai2thor.controller import Controller
from attr import define

from procthor.constants import USE_ITHOR_SPLITS
from procthor.utils.types import Split


@define(eq=False)
class ProcTHORDatabase:
    SOLID_WALL_COLORS: List[Dict[str, float]]
    MATERIAL_DATABASE: Dict[str, List[str]]
    SKYBOXES: Dict[str, Dict[str, str]]
    OBJECTS_IN_RECEPTACLES: Dict[str, Dict[str, Dict[str, Union[float, int]]]]
    ASSET_DATABASE: Dict[str, List[Dict[str, Any]]]
    ASSET_ID_DATABASE: Dict[str, Any]
    PLACEMENT_ANNOTATIONS: pd.DataFrame
    ASSET_GROUPS: Dict[str, Any]
    ASSETS_DF: pd.DataFrame
    WALL_HOLES: Dict[str, Dict[str, Dict[str, float]]]
    FLOOR_ASSET_DICT: Dict[Tuple[str, str], Tuple[Dict[str, Any], pd.DataFrame]]

    PRIORITY_ASSET_TYPES: Dict[
        str, List[str]
    ]  # TODO: Move to sampling parameters in some way?
    """These objects should be placed first inside of the rooms."""


@lru_cache(maxsize=None)
def get_spawnable_asset_group_info(
    split: Split, controller: Controller, pt_db: ProcTHORDatabase
) -> pd.DataFrame:
    from procthor.generation.asset_groups import AssetGroupGenerator

    asset_groups = pt_db.ASSET_GROUPS
    asset_database = pt_db.ASSET_DATABASE

    data = []
    for asset_group_name, asset_group_data in asset_groups.items():
        asset_group_generator = AssetGroupGenerator(
            name=asset_group_name,
            split=split,
            data=asset_group_data,
            controller=controller,
            pt_db=pt_db,
        )

        dims = asset_group_generator.dimensions
        group_properties = asset_group_data["groupProperties"]

        # NOTE: This is kinda naive, since a single asset in the asset group
        # could map to multiple different types of asset types (e.g., Both Chair
        # and ArmChair could be in the same asset).
        # NOTE: use the asset_group_generator.data instead of asset_group_data
        # since it only includes assets from a given split.
        asset_types_in_group = set(
            asset_type
            for asset in asset_group_generator.data["assetMetadata"].values()
            for asset_type, asset_id in asset["assetIds"]
        )
        group_data = {
            "assetGroupName": asset_group_name,
            "assetGroupGenerator": asset_group_generator,
            "xSize": dims["x"],
            "ySize": dims["y"],
            "zSize": dims["z"],
            "inBathrooms": group_properties["roomWeights"]["bathrooms"],
            "inBedrooms": group_properties["roomWeights"]["bedrooms"],
            "inKitchens": group_properties["roomWeights"]["kitchens"],
            "inLivingRooms": group_properties["roomWeights"]["livingRooms"],
            "allowDuplicates": group_properties["properties"]["allowDuplicates"],
            "inCorner": group_properties["location"]["corner"],
            "onEdge": group_properties["location"]["edge"],
            "inMiddle": group_properties["location"]["middle"],
        }

        # NOTE: Add which types are in this asset group
        for asset_type in asset_database.keys():
            group_data[f"has{asset_type}"] = asset_type in asset_types_in_group

        data.append(group_data)

    return pd.DataFrame(data)


class DatabaseLoader:
    def __init__(self, databases_dir: Optional[str] = None):
        self.databases_dir = databases_dir

    def _load_json_from_database(self, json_file: str) -> Union[list, dict]:
        filepath = os.path.join(self.databases_dir, json_file)
        with open(filepath, "r") as f:
            return json.load(f)

    def get_solid_wall_colors(
        self,
    ) -> List[Dict[str, float]]:
        return self._load_json_from_database("solid-wall-colors.json")

    def get_material_database(
        self,
    ) -> dict:
        return self._load_json_from_database("material-database.json")

    def get_asset_database(
        self,
    ) -> dict:
        return (
            self._load_json_from_database("procthor-ithor-split-asset-database.json")
            if USE_ITHOR_SPLITS
            else self._load_json_from_database("asset-database.json")
        )

    def get_skyboxes(
        self,
    ) -> Dict[str, Dict[str, str]]:
        return self._load_json_from_database("skyboxes.json")

    def get_asset_id_database(
        self,
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        asset_type_database = self.get_asset_database()
        asset_id_database = dict()
        for assets in asset_type_database.values():
            for asset in assets:
                asset_id_database[asset["assetId"]] = asset
        return asset_id_database

    def get_ai2thor_object_metadata(
        self,
    ) -> dict:
        return self._load_json_from_database("ai2thor-object-metadata.json")

    def get_placement_annotations(self, fillna: bool = True) -> pd.DataFrame:
        filepath = os.path.join(self.databases_dir, "placement-annotations.json")

        df = pd.read_json(filepath)
        if fillna:
            for key in ["inKitchens", "inLivingRooms", "inBedrooms", "inBathrooms"]:
                df[key] = df[key].fillna(0).astype(np.uint8)
            for key in [
                "inCorner",
                "inMiddle",
                "onEdge",
                "onFloor",
                "onWall",
                "isPickupable",
                "isKinematic",
                "isStructure",
                "multiplePerRoom",
            ]:
                df[key] = df[key].fillna(0).astype(bool)
        df.index.names = ["assetType"]
        return df

    def get_asset_groups(
        self,
    ) -> Dict[str, Any]:
        """Maps each asset group to the web metadata of the asset group."""
        out = dict()
        dirname = os.path.join(self.databases_dir, "asset_groups")
        for fname in os.listdir(dirname):
            if not fname.endswith(".json"):
                continue
            with open(f"{dirname}/{fname}", "r") as f:
                out[fname[: -len(".json")]] = json.load(f)
        return out

    def get_object_in_receptacles(
        self,
    ) -> Dict[str, Dict[str, Dict[str, Union[float, int]]]]:
        return self._load_json_from_database("receptacles.json")

    def get_assets_df(
        self,
    ) -> pd.DataFrame:
        asset_id_db = self.get_asset_id_database()
        return pd.DataFrame(
            [
                {
                    "split": asset["split"],
                    "assetId": asset["assetId"],
                    "objectType": asset["objectType"],
                    "xSize": asset["boundingBox"]["x"],
                    "ySize": asset["boundingBox"]["y"],
                    "zSize": asset["boundingBox"]["z"],
                }
                for asset in asset_id_db.values()
            ]
        )

    def get_wall_holes(
        self,
    ) -> Dict[str, Any]:
        return self._load_json_from_database("wall-holes.json")


def _get_floor_assets(
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
    assets = assets[assets["split"].isin([split, None])]
    assets.set_index("assetId", inplace=True)
    return floor_types, assets


def _get_default_floor_assets_from_key(key: Tuple[str, str]):
    return _get_floor_assets(*key, pt_db=DEFAULT_PROCTHOR_DATABASE)


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


_DDL = DatabaseLoader(databases_dir=os.path.dirname(__file__))

DEFAULT_PROCTHOR_DATABASE = ProcTHORDatabase(
    SOLID_WALL_COLORS=_DDL.get_solid_wall_colors(),
    MATERIAL_DATABASE=_DDL.get_material_database(),
    SKYBOXES=_DDL.get_skyboxes(),
    OBJECTS_IN_RECEPTACLES=_DDL.get_object_in_receptacles(),
    ASSET_DATABASE=_DDL.get_asset_database(),
    ASSET_ID_DATABASE=_DDL.get_asset_id_database(),
    PLACEMENT_ANNOTATIONS=_DDL.get_placement_annotations(),
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
