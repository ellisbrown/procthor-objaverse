import copy
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any

import compress_json
import pandas as pd
import prior
import tqdm
from procthor.databases import DEFAULT_PROCTHOR_DATABASE, ProcTHORDatabase

from procthor.constants import USE_ITHOR_SPLITS, PROCESSED_ASSET_DIRECTORY
from procthor.objaverse.objaverse_constants import OBJAVERSE_DATASETS_DIR

OBJAVERSE_DIR = os.path.abspath(os.path.dirname(Path(__file__)))

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
    assert USE_ITHOR_SPLITS
    asset_database = pt_db.ASSET_DATABASE

    for assets in pt_db.ASSET_DATABASE.values():
        for asset in assets:
            asset["isObjaverse"] = False
            asset["refObjectType"] = None

    for asset in annotations.values():
        object_type = asset["object_type"]

        if object_type not in asset_database:
            asset_database[object_type] = []

        asset_database[object_type].append(
            {
                "assetId": asset["uid"],
                "boundingBox": asset["boundingBox"],
                "materials": [],
                "maxImagePixelLength": None,  # TODO: Don't think we need this
                "objectType": object_type,
                "scenes": [],  # TODO: Don't think we need this
                "primaryProperty": asset["primaryProperty"],
                "secondaryProperties": asset["secondaryProperties"],
                "split": asset["split"],
                "states": {},  # TODO: This seems to only apply to objects that open
                "isObjaverse": True,
                "refObjectType": asset["ref_category"],
            }
        )

    with open(os.path.join(OBJAVERSE_DATASETS_DIR, "asset-database.json"), "w") as f:
        json.dump(asset_database, f, indent=4)

    with open(
        os.path.join(
            OBJAVERSE_DATASETS_DIR, "procthor-ithor-split-asset-database.json"
        ),
        "w",
    ) as f:
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
    start_len = placement_df.shape[0]

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

    pt_db.PLACEMENT_ANNOTATIONS = pd.concat((placement_df, pd.DataFrame(new_rows)))
    pt_db.PLACEMENT_ANNOTATIONS["isObjaverse"] = [False] * start_len + [True] * len(
        new_rows
    )
    pt_db.PLACEMENT_ANNOTATIONS.to_json(
        os.path.join(OBJAVERSE_DATASETS_DIR, "placement-annotations.json"), indent=4
    )


def create_asset_groups(
    pt_db: ProcTHORDatabase, annotations: Dict[str, Dict[str, Any]]
):
    # TODO: Do we want to use objaverse assets for asset groups?
    asset_groups = pt_db.ASSET_GROUPS

    asset_group_dir = os.path.join(OBJAVERSE_DATASETS_DIR, "asset_groups")
    os.makedirs(asset_group_dir, exist_ok=True)
    for asset_group, group_info in asset_groups.items():
        with open(os.path.join(asset_group_dir, f"{asset_group}.json"), "w") as f:
            json.dump(group_info, f, indent=4)


def create_receptacles_database(
    pt_db: ProcTHORDatabase, annotations: Dict[str, Dict[str, Any]]
):
    contained_to_receptacles = defaultdict(dict)
    for receptacle, contained_to_info in pt_db.OBJECTS_IN_RECEPTACLES.items():
        for contained, info in contained_to_info.items():
            contained_to_receptacles[contained][receptacle] = copy.deepcopy(info)

    ot_to_ref_ot = {v["object_type"]: v["ref_category"] for v in annotations.values()}

    new_receptacle_to_contained = copy.deepcopy(pt_db.OBJECTS_IN_RECEPTACLES)
    for ot, ref_ot in ot_to_ref_ot.items():
        if ref_ot in contained_to_receptacles:
            for containing_recepts, info in contained_to_receptacles[ref_ot].items():
                new_receptacle_to_contained[containing_recepts][ot] = copy.deepcopy(
                    info
                )

    for ot, ref_ot in ot_to_ref_ot.items():
        if ref_ot in new_receptacle_to_contained:
            new_receptacle_to_contained[ot] = copy.deepcopy(
                new_receptacle_to_contained[ref_ot]
            )

    pt_db.OBJECTS_IN_RECEPTACLES = new_receptacle_to_contained

    with open(os.path.join(OBJAVERSE_DATASETS_DIR, "receptacles.json"), "w") as f:
        json.dump(pt_db.OBJECTS_IN_RECEPTACLES, f, indent=2)


def create_skyboxes_database(
    pt_db: ProcTHORDatabase, annotations: Dict[str, Dict[str, Any]]
):
    with open(os.path.join(OBJAVERSE_DATASETS_DIR, "skyboxes.json"), "w") as f:
        json.dump(pt_db.SKYBOXES, f, indent=2)


def create_solid_wall_colors_database(
    pt_db: ProcTHORDatabase, annotations: Dict[str, Dict[str, Any]]
):
    with open(os.path.join(OBJAVERSE_DATASETS_DIR, "solid-wall-colors.json"), "w") as f:
        json.dump(pt_db.SOLID_WALL_COLORS, f, indent=2)


def create_wall_holes_database(
    pt_db: ProcTHORDatabase, annotations: Dict[str, Dict[str, Any]]
):
    with open(os.path.join(OBJAVERSE_DATASETS_DIR, "wall-holes.json"), "w") as f:
        json.dump(pt_db.WALL_HOLES, f, indent=2)


def objaverse_cat_to_thor_cat(objaverse_cat: str):
    objaverse_cat = re.sub(r"[^a-zA-Z0-9 ]", " ", objaverse_cat)
    return "Obja" + objaverse_cat.title().replace(" ", "")


def refine_annotations_and_update_thor_metadata(
    pt_db: ProcTHORDatabase,
    annotations: Dict[str, Dict[str, Any]],
    path_to_objects: str,
    overwrite: bool = False,
):
    refined_annotations = {}
    missing_metadata = []

    allowed_secondary_properties = [
        "Receptacle",
        "CanPickup",
        "Moveable",
    ]  # Don't allow things like "CanOpen"

    for oid in tqdm.tqdm(annotations, desc="Refining annotations"):
        md_path = os.path.join(path_to_objects, oid, "thor_metadata.json")
        if not os.path.exists(md_path):
            missing_metadata.append(oid)
            continue
        with open(md_path, "r") as f:
            md = json.load(f)

        object_annotation = copy.deepcopy(annotations[oid])
        refined_annotations[oid] = object_annotation
        object_annotation["boundingBox"] = md["objectMetadata"][
            "axisAlignedBoundingBox"
        ]["size"]

        object_annotation["object_type"] = objaverse_cat_to_thor_cat(
            object_annotation["category"]
        )

        if overwrite or md["assetMetadata"]["primaryProperty"] in [
            "Undefined",
            "Static",
        ]:
            ref_ot = annotations[oid]["ref_category"]
            ref_obj = pt_db.ASSET_DATABASE[ref_ot][0]

            max_dim = max(object_annotation["boundingBox"].values())

            if max_dim <= 0.5:
                md["assetMetadata"]["primaryProperty"] = "CanPickup"
            else:
                md["assetMetadata"]["primaryProperty"] = ref_obj.get(
                    "primaryProperty", "Moveable"
                )
                md["assetMetadata"]["secondaryProperties"] = [
                    sp
                    for sp in ref_obj.get("secondaryProperties", [])
                    if sp in allowed_secondary_properties
                ]

            with open(md_path, "w") as f:
                json.dump(md, f, indent=2)

        object_annotation["primaryProperty"] = md["assetMetadata"]["primaryProperty"]
        object_annotation["secondaryProperties"] = md["assetMetadata"][
            "secondaryProperties"
        ]
        object_annotation["pickupable"] = (
            md["assetMetadata"]["primaryProperty"] == "CanPickup"
        )

        object_annotation["filesize"] = int(
            os.path.getsize(os.path.join(path_to_objects, oid, f"{oid}.pkl.gz"))
        )

    print(
        f"New annotations contain {len(refined_annotations)} objects."
        f" Metadata missing for for {len(missing_metadata)} objects.",
        flush=True,
    )
    return refined_annotations


def filter_annotations(
    annotations: Dict[str, Dict[str, Any]],
):
    filtered = {}

    for oid, info in tqdm.tqdm(annotations.items(), desc="Filtering annotations"):
        bb = info["boundingBox"]
        pickupable = info["pickupable"]
        max_dim = max(bb.values())
        y_size = bb["y"]

        reject = False

        if pickupable:
            if not (0.05 <= max_dim <= 2):
                reject = True
        else:
            if not (0.05 <= max_dim <= 3 and y_size < 2.5):
                reject = True

        reject = reject or (info["object_type"] == "ObjaToilet")
        reject = reject or (info["filesize"] > 1.5 * (2**20))  # Reject files > 1.5mb

        reject = reject or (
            info["ref_category"] == "Painting"
            and (bb["z"] > 0.4 or bb["z"] > min(bb["x"], bb["y"]))
        )

        if not reject:
            filtered[oid] = info

    print(
        f"Filtered annotations contain {len(filtered)} objects out of {len(annotations)}"
    )
    return filtered


def create_hash_seed(s: str) -> int:
    h = hashlib.md5()
    h.update(s.encode())
    return int(h.hexdigest(), 16) % 2**31


def create_splits(
    annotations: Dict[str, Dict[str, Any]],
):
    cat_to_objects = defaultdict(list)

    for oid, info in annotations.items():
        cat_to_objects[info["category"]].append(info)

    for cat, objects in tqdm.tqdm(cat_to_objects.items(), "Creating splits"):
        objects.sort(key=lambda x: x["uid"])
        random.Random(create_hash_seed(cat)).shuffle(objects)

        ntest = max(math.floor(len(objects) * 0.1), 1)
        ntrain = min(round(len(objects) * 0.8), max(len(objects) - ntest, 0))
        nvalid = len(objects) - ntest - ntrain

        for obj in objects[:ntrain]:
            obj["split"] = "train"

        for obj in objects[ntrain : ntrain + nvalid]:
            obj["split"] = "val"

        for obj in objects[ntrain + nvalid :]:
            obj["split"] = "test"

    return annotations


def print_annotations_summary(annotations: Dict[str, Dict[str, Any]]):
    df = pd.DataFrame(annotations.values())
    df["is_train"] = df["split"] == "train"
    df["is_val"] = df["split"] == "val"
    df["is_test"] = df["split"] == "test"

    print(
        f"Annotations contain {len(df)} objects, {len(df[df['is_train']])} train, {len(df[df['is_val']])} val, {len(df[df['is_test']])} test"
    )
    summed = df.groupby("category")
    print(
        f"{(summed['is_train'].sum() > 0).sum()} train categories ({df['is_train'].sum()} objects)"
    )
    print(
        f"{(summed['is_val'].sum() > 0).sum()} val categories ({df['is_val'].sum()} objects)"
    )
    print(
        f"{(summed['is_test'].sum() > 0).sum()} test categories ({df['is_test'].sum()} objects)"
    )


def remove_categories_that_are_not_in_train(annotations: Dict[str, Dict[str, Any]]):
    train_categories = {
        info["category"] for info in annotations.values() if info["split"] == "train"
    }

    new_annotations = {}
    for oid, info in annotations.items():
        if info["category"] in train_categories:
            new_annotations[oid] = info

    return new_annotations


def main(should_create_splits: bool = False):
    annotations = prior.load_dataset(
        "objaverse-plus", revision="4f23a101f2a21debd784210ce568cc1ada9cd913"
    )["train"].data

    new_db = copy.deepcopy(DEFAULT_PROCTHOR_DATABASE)

    if should_create_splits:
        assert all("split" not in info for info in annotations.values())
        annotations = create_splits(annotations)
        compress_json.dump(
            annotations, os.path.join(OBJAVERSE_DATASETS_DIR, "thor_subset.json.gz")
        )
    else:
        assert all(
            info["split"] in ["train", "val", "test"] for info in annotations.values()
        )

    print("\nBefore filtering and refining:")
    print_annotations_summary(annotations)

    annotations = refine_annotations_and_update_thor_metadata(
        pt_db=new_db,
        annotations=annotations,
        path_to_objects=PROCESSED_ASSET_DIRECTORY,
        overwrite=False,
    )

    annotations = filter_annotations(annotations)

    print("\nAfter filtering and refining:")
    print_annotations_summary(annotations)

    annotations = remove_categories_that_are_not_in_train(annotations)

    print("\nAfter removing categories not in train:")
    print_annotations_summary(annotations)

    compress_json.dump(
        annotations, os.path.join(OBJAVERSE_DATASETS_DIR, "refined_annotations.json")
    )

    # import matplotlib.pyplot as plt
    # import numpy as np

    # df = pd.DataFrame(annotations.values())
    # df["is_train"] = df["split"] == "train"
    # df["is_val"] = df["split"] == "val"
    # df["is_test"] = df["split"] == "test"
    #
    # print(
    #     f"Contains {len(df)} objects, {len(df[df['is_train']])} train, {len(df[df['is_val']])} val, {len(df[df['is_test']])} test"
    # )
    # summed = df.groupby("category")
    # print(f"{(summed['is_train'].sum() > 0).sum()} train categories")
    # print(f"{(summed['is_val'].sum() > 0).sum()} val categories")
    # print(f"{(summed['is_test'].sum() > 0).sum()} test categories")

    # df["volume"] = [
    #     np.product(list(a["boundingBox"].values())) for a in annotations.values()
    # ]
    # df["max_dim"] = [max(list(a["boundingBox"].values())) for a in annotations.values()]
    # pdf = df[df["pickupable"]]
    # mdf = df[~df["pickupable"]]
    # # plot histogram of values
    # plt.hist(df["volume"][df["volume"] < 6.9], bins=50)
    # np.percentile(df[df["pickupable"]]["max_dim"], [93, 98, 99, 100])
    # plt.hist(df[~df["pickupable"]]["max_dim"], bins=50)
    # plt.hist(df["filesize"], bins=50)
    # plt.show()

    # update_thor_metadata_with_primary_and_secondary_properties(
    #     pt_db=new_db, annotations=annotations, path_to_objects=PROCESSED_ASSET_DIRECTORY
    # )

    create_asset_database(pt_db=new_db, annotations=annotations)
    create_material_database(pt_db=new_db, annotations=annotations)
    create_placement_annotations(pt_db=new_db, annotations=annotations)
    # create_object_groups(pt_db=new_db, annotations=annotations) # Not needed
    create_asset_groups(pt_db=new_db, annotations=annotations)
    create_receptacles_database(pt_db=new_db, annotations=annotations)
    create_skyboxes_database(pt_db=new_db, annotations=annotations)
    create_skyboxes_database(pt_db=new_db, annotations=annotations)
    create_solid_wall_colors_database(pt_db=new_db, annotations=annotations)
    create_wall_holes_database(pt_db=new_db, annotations=annotations)


if __name__ == "__main__":
    main(should_create_splits=False)
