import copy
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from procthor.databases import DEFAULT_PROCTHOR_DATABASE, ProcTHORDatabase

OBJAVERSE_DIR = os.path.abspath(os.path.dirname(Path(__file__)))
OBJAVERSE_DATASETS_DIR = os.path.join(OBJAVERSE_DIR, "objaverse_databases")

DEFAULT_DECOR_PLACEMENT = {
    "instances": 1,
    "inKitchens": 1,
    "inLivingRooms": 1,
    "inBedrooms": 1,
    "inBathrooms": 0,
    "inCorner": False,
    "inMiddle": False,
    "onEdge": False,
    "onFloor": False,
    "onWall": False,
    "isPickupable": True,
    "isKinematic": False,
    "isStructure": False,
    "multiplePerRoom": True,
}


def create_asset_database(
    pt_db: ProcTHORDatabase, annotations: Dict[str, Dict[str, Any]]
):
    asset_database = pt_db.ASSET_DATABASE
    for asset in annotations.values():
        object_type = asset["object_type"]

        if object_type not in asset_database:
            asset_database[object_type] = []

        asset_database[object_type].append(
            {
                "assetId": asset["uid"],
                "boundingBox": {},  # TODO
                "materials": [],
                "maxImagePixelLength": None,  # TODO: Don't think we need this
                "objectType": object_type,
                "scenes": [],  # TODO: Don't think we need this
                "secondaryProperties": [],
                "split": "train",  # TODO
                "states": {},  # TODO: This seems to only apply to objects that open
            }
        )

    with open(os.path.join(OBJAVERSE_DATASETS_DIR, "asset-database.json"), "w") as f:
        json.dump(asset_database, f, indent=4)


def create_material_database(
    pt_db: ProcTHORDatabase, annotations: Dict[str, Dict[str, Any]]
):
    material_database = pt_db.MATERIAL_DATABASE

    object_types = sorted(list(set(v["object_type"] for v in annotations.values())))

    for ot in object_types:
        material_database[ot] = []

    with open(os.path.join(OBJAVERSE_DATASETS_DIR, "material-database.json"), "w") as f:
        json.dump(material_database, f, indent=4)


def create_placement_annotations(
    pt_db: ProcTHORDatabase, annotations: Dict[str, Dict[str, Any]]
):
    placement_df = pt_db.PLACEMENT_ANNOTATIONS

    ot_to_ref_ot = {v["object_type"]: v["ref_category"] for v in annotations.values()}
    ot_to_count = Counter(v["object_type"] for v in annotations.values())

    new_rows = []
    for ot, ref_ot in sorted(ot_to_ref_ot.items()):
        if ref_ot in ["RoomDecor", "TableTopDecor"]:
            new_row = copy.deepcopy(DEFAULT_DECOR_PLACEMENT)
        else:
            new_row = copy.deepcopy(dict(placement_df.loc[ref_ot]))

        new_row["instances"] = ot_to_count[ot]
        s = pd.Series(new_row)
        s.name = ot
        new_rows.append(s)

    pt_db.PLACEMENT_ANNOTATIONS = placement_df.append(pd.DataFrame(new_rows))
    pt_db.PLACEMENT_ANNOTATIONS.to_json(
        os.path.join(OBJAVERSE_DATASETS_DIR, "placement-annotations.json"), indent=4
    )


def objaverse_cat_to_thor_cat(objaverse_cat: str):
    objaverse_cat = re.sub(r"[^a-zA-Z0-9 ]", " ", objaverse_cat)
    return "Obja" + objaverse_cat.title().replace(" ", "")


def main():
    with open(
        os.path.join(OBJAVERSE_DIR, "annotations/objaverse_thor_vp0p9.json"), "r"
    ) as f:
        annotations = json.load(f)

    for v in annotations.values():
        v["object_type"] = objaverse_cat_to_thor_cat(v["category"])

    new_db = copy.deepcopy(DEFAULT_PROCTHOR_DATABASE)

    create_asset_database(pt_db=new_db, annotations=annotations)
    create_material_database(pt_db=new_db, annotations=annotations)
    create_placement_annotations(pt_db=new_db, annotations=annotations)


if __name__ == "__main__":
    main()
