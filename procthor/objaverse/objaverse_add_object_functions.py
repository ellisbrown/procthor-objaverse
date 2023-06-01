import copy
import logging
import pdb
import random
import sys
import warnings
from collections import defaultdict, Counter
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Callable,
)

import numpy as np
import pandas as pd
from procthor.databases import ProcTHORDatabase, get_spawnable_asset_group_info
from shapely.geometry import MultiLineString

from ai2thor.controller import Controller
from procthor.constants import (
    OPENNESS_RANDOMIZATIONS,
    FLOOR_Y,
)
from procthor.generation import PartialHouse
from procthor.generation.objects import (
    P_ALLOW_HOUSE_PLANT_GROUP,
    P_ALLOW_TV_GROUP,
    sample_and_add_floor_asset,
    AssetGroup,
    ProceduralRoom,
    sample_openness,
)
from procthor.generation.small_objects import (
    randomize_bias,
    PARENT_BIAS,
    CHILD_BIAS,
    MAX_OF_TYPE_ON_RECEPTACLE,
    HOUSE_PLANT_MAX_HEIGHT,
    OBJECTS_TO_DROP,
    FLOOR_OBJECTS_TO_DROP,
)
from procthor.generation.wall_objects import (
    add_windows,
    add_televisions,
    add_paintings,
    filter_room_lines_df,
    get_wall_placement_info,
    sample_asset_y_position,
)
from procthor.objaverse.objaverse_constants import (
    MIN_OBJAVERSE_INSTANCES_FOR_HEAD_CATEGORY,
    MAX_HEAD_OBJAVERSE_OBJECT_TYPES_PER_ROOM,
    MAX_TAIL_OBJAVERSE_OBJECT_TYPES_PER_ROOM,
    EXCLUDE_NON_OBJAVERSE_ASSETS,
    OBJAVERSE_WALL_OBJECTS_PER_ROOM,
    ALLOW_DUPLICATE_OBJAVERSE_WALL_OBJECTS_IN_HOUSE,
)
from procthor.utils.types import Object, Split, Vector3, BoundaryGroups, Wall


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def split_to_split_choices(split: str) -> List[str]:
    split_choices = ["train"]
    if split == "train":
        pass
    elif split == "val":
        split_choices.append("val")
    elif split == "test":
        split_choices.append("test")
    else:
        raise NotImplementedError(f"Unknown split {split}")

    return split_choices


def objaverse_add_floor_objects(
    partial_house: PartialHouse,
    controller: Controller,
    pt_db: ProcTHORDatabase,
    split: Split,
    max_floor_objects: int,
    p_allow_house_plant_group: float = P_ALLOW_HOUSE_PLANT_GROUP,
    p_allow_tv_group: float = P_ALLOW_TV_GROUP,
) -> None:
    """Add objects to each room.

    Args:
        floor_polygons: Maps each room's id to the shapely polygon of each room's
            floor.

    """
    assert partial_house.objects is None or len(partial_house.objects) == 0

    partial_house.objects = []
    for room in partial_house.rooms.values():
        allow_house_plant_group = random.random() < p_allow_house_plant_group
        allow_tv_group = random.random() < p_allow_tv_group

        priority_asset_types = copy.deepcopy(pt_db.PRIORITY_ASSET_TYPES[room.room_type])
        random.shuffle(priority_asset_types)

        split_choices = split_to_split_choices(split=room.split)

        random.shuffle(split_choices)

        spawnable_assets_df_list = []
        for split in split_choices:
            _, spawnable_assets = pt_db.FLOOR_ASSET_DICT[(room.room_type, split)]

            if EXCLUDE_NON_OBJAVERSE_ASSETS:
                spawnable_assets = spawnable_assets[
                    [
                        at.startswith("Obja") or at in priority_asset_types
                        for at in spawnable_assets["assetType"]
                    ]
                ]
            spawnable_assets_df_list.append(spawnable_assets)
        assert any(sa_df.shape[0] > 0 for sa_df in spawnable_assets_df_list)
        spawnable_assets_df_list = [
            sa_df for sa_df in spawnable_assets_df_list if sa_df.shape[0] > 0
        ]

        spawnable_asset_group_info = get_spawnable_asset_group_info(
            splits=tuple(split_choices), controller=controller, pt_db=pt_db
        )
        spawnable_asset_groups = spawnable_asset_group_info[
            spawnable_asset_group_info[f"in{room.room_type}s"] > 0
        ]

        asset = None
        for i in range(max_floor_objects):
            cache_rectangles = i != 0 and asset is None
            if cache_rectangles:
                # NOTE: Don't resample failed rectangles
                room.last_rectangles.remove(rectangle)
                rectangle = room.sample_next_rectangle(cache_rectangles=True)
            else:
                rectangle = room.sample_next_rectangle()

            if rectangle is None:
                break

            x_info, z_info, anchor_delta, anchor_type = room.sample_anchor_location(
                rectangle
            )

            asset = sample_and_add_floor_asset(
                room=room,
                rectangle=rectangle,
                anchor_type=anchor_type,
                anchor_delta=anchor_delta,
                allow_house_plant_group=allow_house_plant_group,
                allow_tv_group=allow_tv_group,
                spawnable_assets=random.choice(spawnable_assets_df_list),
                spawnable_asset_groups=spawnable_asset_groups,
                priority_asset_types=priority_asset_types,
                pt_db=pt_db,
            )
            # NOTE: no asset within the asset group could be placed inside of the
            # rectangle.
            if asset is None:
                continue

            room.sample_place_asset_in_rectangle(
                asset=asset,
                rectangle=rectangle,
                anchor_type=anchor_type,
                x_info=x_info,
                z_info=z_info,
                anchor_delta=anchor_delta,
            )

            added_asset_types = []
            if "assetType" in asset:
                added_asset_types.append(asset["assetType"])
            else:
                added_asset_types.extend([o["assetType"] for o in asset["objects"]])

                if not asset["allowDuplicates"]:
                    spawnable_asset_groups = spawnable_asset_groups.query(
                        f"assetGroupName!='{asset['assetGroupName']}'"
                    )

            for asset_type in added_asset_types:
                # Remove spawned object types from `priority_asset_types` when appropriate
                if asset_type in priority_asset_types:
                    priority_asset_types.remove(asset_type)

                allow_duplicates_of_asset_type = pt_db.PLACEMENT_ANNOTATIONS.loc[
                    asset_type
                ]["multiplePerRoom"]

                if not allow_duplicates_of_asset_type:
                    # NOTE: Remove all asset groups that have the type
                    spawnable_asset_groups = spawnable_asset_groups[
                        ~spawnable_asset_groups[f"has{asset_type}"]
                    ]

                    # NOTE: Remove all standalone assets that have the type
                    spawnable_assets_df_list = [
                        sd_df[sd_df["assetType"] != asset_type]
                        for sd_df in spawnable_assets_df_list
                    ]

        # NOTE: add the formatted assets
        for asset in room.assets:
            if isinstance(asset, AssetGroup):
                partial_house.objects.extend(asset.assets_dict)
            else:
                partial_house.objects.append(asset.asset_dict)


def get_objaverse_assets_df(
    pt_db: ProcTHORDatabase,
    split: str,
    asset_filter: Optional[Callable[[Dict[str, Any]], bool]],
) -> pd.DataFrame:
    split_list = [split]
    if split == "train":
        split_list.append(None)

    return pd.DataFrame(
        [
            {
                "assetId": asset["assetId"],
                "xSize": asset["boundingBox"]["x"],
                "ySize": asset["boundingBox"]["y"],
                "zSize": asset["boundingBox"]["z"],
                "split": asset["split"] if asset["split"] is not None else "train",
                "objectType": asset["objectType"],
            }
            for assets in pt_db.ASSET_DATABASE.values()
            for asset in assets
            if (split == "all" or asset["split"] in split_list)
            and asset["objectType"].startswith("Obja")
            and (asset_filter is None or asset_filter(asset))
        ]
    )


def add_objaverse_wall_objects(
    partial_house: PartialHouse,
    rooms: Dict[int, ProceduralRoom],
    split: Split,
    wall_map: Dict[str, Wall],
    ceiling_height: float,
    tvs_per_room: int,
    rooms_lines_df_map,
    wall_object_heights_per_room,
    pt_db: ProcTHORDatabase,
) -> None:
    """Add paintings to the house."""
    split_choices = split_to_split_choices(split=split)

    random.shuffle(split_choices)

    joint_objaverse_df = get_objaverse_assets_df(
        pt_db=pt_db,
        split="all",
        asset_filter=lambda asset: asset["refObjectType"] == "Painting",
    )

    max_objaverse_wall_objects_in_rooms = random.choices(
        k=len(rooms_lines_df_map), **OBJAVERSE_WALL_OBJECTS_PER_ROOM
    )

    min_obja_size = joint_objaverse_df["xSize"].min()
    for max_in_room, (room_id, room_lines_df) in zip(
        max_objaverse_wall_objects_in_rooms, rooms_lines_df_map.items()
    ):
        for obja_i in range(max_in_room):
            room_lines_df = filter_room_lines_df(
                room_lines_df=room_lines_df, min_asset_size=min_obja_size
            )

            # NOTE: No more space on the walls
            if (not len(room_lines_df)) or (joint_objaverse_df.shape == 0):
                break

            # NOTE: sample the line string
            room_line_i = random.choices(
                population=list(room_lines_df.index),
                weights=list(room_lines_df["length"]),
                k=1,
            )[0]
            room_line = room_lines_df.loc[room_line_i]

            # NOTE: sample the objaverse object
            random.shuffle(split_choices)
            obja_candidates = None
            for split_choice in split_choices:
                sub_df = joint_objaverse_df[joint_objaverse_df["split"] == split_choice]
                obja_candidates = sub_df[sub_df["xSize"] < room_line["length"]]
                if obja_candidates.shape[0] > 1:
                    break

            # NOTE: No more space on the walls
            if obja_candidates.shape[0] == 0:
                break

            object_to_add = obja_candidates.sample()

            # NOTE: Choose the position of the painting
            start_position = random.random() * (
                room_line["length"] - object_to_add["xSize"].iloc[0]
            )

            wall_poly = wall_map[room_line["wallId"]]["polygon"]

            placement = get_wall_placement_info(
                wall_poly=wall_poly,
                room_line=room_line,
                asset=object_to_add,
                start_position=start_position,
            )

            min_y, max_y = sample_asset_y_position(
                asset_height=object_to_add["ySize"].iloc[0],
                wall_object_heights=wall_object_heights_per_room[room_id],
                asset_top_down_poly=placement["poly"],
                ceiling_height=ceiling_height,
            )
            center_y_position = (min_y + max_y) / 2

            object_type = object_to_add["objectType"].iloc[0]
            partial_house.objects.append(
                Object(
                    id=f"{object_type}|{room_id}|wall|{obja_i}",
                    assetId=object_to_add["assetId"].iloc[0],
                    objectType=object_to_add["objectType"].iloc[0],
                    position=Vector3(
                        x=placement["centerX"],
                        y=center_y_position,
                        z=placement["centerZ"],
                    ),
                    rotation=Vector3(x=0, y=placement["rotation"], z=0),
                    kinematic=True,
                )
            )

            # NOTE: subtract painting from valid locations in room
            room_lines_df = room_lines_df.drop(room_line_i)
            rooms_lines_df_map[room_id] = rooms_lines_df_map[room_id][
                rooms_lines_df_map[room_id]["lineString"] != room_line["lineString"]
            ]

            line_string = room_line["lineString"]
            line_string -= placement["poly"]

            if isinstance(line_string, MultiLineString):
                line_strings_to_add = [
                    ls for ls in line_string.geoms if ls.length > min_obja_size
                ]
            elif line_string.length > min_obja_size:
                line_strings_to_add = [line_string]
            else:
                line_strings_to_add = []

            if line_strings_to_add:
                lines_to_append = []
                for line_string in line_strings_to_add:
                    start, end = line_string.boundary.geoms
                    x1, z1 = start.x, start.y
                    x2, z2 = end.x, end.y
                    lines_to_append.append(
                        {
                            "length": line_string.length,
                            "x1": min(x1, x2),
                            "x2": max(x1, x2),
                            "z1": min(z1, z2),
                            "z2": max(z1, z2),
                            "lineString": line_string,
                            "wallId": room_line["wallId"],
                        }
                    )
                lines_to_append = pd.DataFrame(lines_to_append)
                room_lines_df = pd.concat(
                    [room_lines_df, lines_to_append], ignore_index=True
                )
                rooms_lines_df_map[room_id] = pd.concat(
                    [rooms_lines_df_map[room_id], lines_to_append], ignore_index=True
                )

            # NOTE: Don't allow the same painting to be spawned in.
            if not ALLOW_DUPLICATE_OBJAVERSE_WALL_OBJECTS_IN_HOUSE:
                joint_objaverse_df = joint_objaverse_df.drop(object_to_add.index)
                min_obja_size = joint_objaverse_df["xSize"].min()


def objaverse_add_wall_objects(
    partial_house: PartialHouse,
    controller: Controller,
    pt_db: ProcTHORDatabase,
    split: Split,
    rooms: Dict[int, ProceduralRoom],
    boundary_groups: BoundaryGroups,
    room_type_map: Dict[int, str],
    ceiling_height: float,
) -> None:
    """Add wall objects to the house."""
    wall_map = {w["id"]: w for w in partial_house.walls}
    add_windows(
        partial_house=partial_house,
        rooms=rooms,
        boundary_groups=boundary_groups,
        split=split,
        room_type_map=room_type_map,
        wall_map=wall_map,
        ceiling_height=ceiling_height,
        pt_db=pt_db,
    )
    tvs_per_room, rooms_lines_df_map, wall_object_heights_per_room = add_televisions(
        partial_house=partial_house,
        rooms=rooms,
        boundary_groups=boundary_groups,
        split=split,
        room_type_map=room_type_map,
        wall_map=wall_map,
        ceiling_height=ceiling_height,
        pt_db=pt_db,
    )

    add_paintings(
        partial_house=partial_house,
        rooms=rooms,
        split=split,
        wall_map=wall_map,
        ceiling_height=ceiling_height,
        tvs_per_room=tvs_per_room,
        rooms_lines_df_map=rooms_lines_df_map,
        wall_object_heights_per_room=wall_object_heights_per_room,
        pt_db=pt_db,
    )

    add_objaverse_wall_objects(
        partial_house=partial_house,
        rooms=rooms,
        split=split,
        wall_map=wall_map,
        ceiling_height=ceiling_height,
        tvs_per_room=tvs_per_room,
        rooms_lines_df_map=rooms_lines_df_map,
        wall_object_heights_per_room=wall_object_heights_per_room,
        pt_db=pt_db,
    )


# NOTE: Drop object from near ceiling so it falls
def _set_drop_heights(
    changed_ids: Set[str],
    objects: List[Object],
    object_type_to_prob_of_drop_change: Dict[str, Dict[str, float]],
    object_type_to_prob_of_drop_change_children: Dict[str, Dict[str, float]],
):
    for obj in objects:
        obj_type = obj["objectType"]
        if (
            obj_type in object_type_to_prob_of_drop_change
            and random.random() < object_type_to_prob_of_drop_change[obj_type]["p"]
        ):
            # TODO: This code seems bad to me
            obj["position"]["y"] = 3
            obj["rotation"]["x"] = random.random() * 2 + 3
            obj["rotation"]["y"] = random.random() * 360
            changed_ids.add(obj["id"])

        if "children" in obj:
            # TODO: Intentionally using `object_type_to_prob_of_drop_change_children`
            #   for both `object_type_to_prob_of_drop_change` and `object_type_to_prob_of_drop_change_children`
            #   below. Not sure why this is the case (copied logic from original code).
            _set_drop_heights(
                changed_ids=changed_ids,
                objects=obj["children"],
                object_type_to_prob_of_drop_change=object_type_to_prob_of_drop_change_children,
                object_type_to_prob_of_drop_change_children=object_type_to_prob_of_drop_change_children,
            )


# NOTE: Get pose of dropped object
def _save_new_heights(
    changed_ids: Set[str],
    partial_house_objects: List[Object],
    thor_objects: List[Dict[str, Any]],
):
    for obj in partial_house_objects:
        if obj["id"] in changed_ids:
            thor_obj = next(o for o in thor_objects if o["objectId"] == obj["id"])
            obj["position"] = thor_obj["axisAlignedBoundingBox"]["center"].copy()
            obj["rotation"] = thor_obj["rotation"].copy()
        if "children" in obj:
            _save_new_heights(
                changed_ids=changed_ids,
                partial_house_objects=obj["children"],
                thor_objects=thor_objects,
            )


def objaverse_add_small_objects(
    partial_house: PartialHouse,
    controller: Controller,
    pt_db: ProcTHORDatabase,
    split: Split,
    rooms: Dict[int, ProceduralRoom],
    max_object_types_per_room: int = 10000,
) -> None:
    """Add small objects to the house."""
    controller.reset()
    controller.step(action="ResetObjectFilter")
    event = controller.step(
        action="CreateHouse", house=partial_house.to_house_dict(), renderImage=False
    )
    assert event, "Unable to CreateHouse!"
    controller.step(action="SetObjectFilter", objectIds=[])

    # NOTE: Get the objects in the room.
    objects = [
        obj
        for obj in event.metadata["objects"]
        if not any(
            obj["objectId"].startswith(k)
            for k in ["wall|", "room|", "Floor", "door|", "window|"]
        )
    ]

    objects_per_room = defaultdict(list)
    for obj in objects:
        object_id = obj["objectId"]
        room_id = int(object_id.split("|")[1])
        objects_per_room[room_id].append(obj)
    objects_per_room = dict(objects_per_room)

    receptacles_per_room = {
        room_id: [
            obj for obj in objects if obj["objectType"] in pt_db.OBJECTS_IN_RECEPTACLES
        ]
        for room_id, objects in objects_per_room.items()
    }

    room_id_to_object_type_to_count_in_room = {
        room_id: dict(Counter(obj["objectType"] for obj in objects))
        for room_id, objects in objects_per_room.items()
    }

    objects_in_house = {obj["id"]: obj for obj in partial_house.objects}

    house_bias = randomize_bias()
    logging.debug(f"Small object bias: {house_bias}")

    # NOTE: Place the objects
    total_small_objects_placed = 0
    for room_id, room in rooms.items():
        if room_id not in receptacles_per_room:
            continue

        object_type_to_count_in_room = room_id_to_object_type_to_count_in_room[room_id]

        receptacles_in_room = receptacles_per_room[room_id]
        room_type = room.room_type

        objaverse_object_types = set(
            pt_db.PLACEMENT_ANNOTATIONS.index[
                pt_db.PLACEMENT_ANNOTATIONS["isObjaverse"]
            ].tolist()
        )

        object_type_to_is_head = {}
        object_type_to_receptacle_infos = defaultdict(list)
        for receptacle in receptacles_in_room:
            objects_in_receptacle = pt_db.OBJECTS_IN_RECEPTACLES[
                receptacle["objectType"]
            ]

            for object_type_to_spawn, data in objects_in_receptacle.items():
                if (
                    EXCLUDE_NON_OBJAVERSE_ASSETS
                    and not object_type_to_spawn.startswith("Obja")
                ):
                    continue

                type_placement_info = pt_db.PLACEMENT_ANNOTATIONS.loc[
                    object_type_to_spawn
                ]
                room_weight = pt_db.PLACEMENT_ANNOTATIONS.loc[object_type_to_spawn][
                    f"in{room_type}s"
                ]
                object_type_to_is_head[object_type_to_spawn] = (
                    type_placement_info["instances"]
                    >= MIN_OBJAVERSE_INSTANCES_FOR_HEAD_CATEGORY
                )

                if room_weight == 0:
                    continue

                if random.random() <= (
                    data["p"]
                    + PARENT_BIAS[receptacle["objectType"]]
                    + CHILD_BIAS[object_type_to_spawn]
                    + house_bias
                ):
                    if data["p"] == 0:
                        multiplicity = 1
                    elif data["p"] == 1:
                        multiplicity = MAX_OF_TYPE_ON_RECEPTACLE
                    else:
                        multiplicity = min(
                            np.random.geometric(p=1 - data["p"], size=1)[0],
                            MAX_OF_TYPE_ON_RECEPTACLE,
                        )

                    object_type_to_receptacle_infos[object_type_to_spawn].append(
                        {
                            "receptacleId": receptacle["objectId"],
                            "receptacleType": receptacle["objectType"],
                            "multiplicity": multiplicity,
                        }
                    )

        objects_types_placed_in_room = set()

        total_head_objaverse_objects_placed = 0
        total_tail_objaverse_objects_placed = 0
        have_placed_multi_objaverse_head_cat = False
        while len(object_type_to_receptacle_infos) != 0:
            if len(objects_types_placed_in_room) >= max_object_types_per_room:
                break

            object_type_to_spawn = random.choice(
                list(object_type_to_receptacle_infos.keys())
            )

            is_objaverse = object_type_to_spawn in objaverse_object_types
            is_head = object_type_to_is_head[object_type_to_spawn]

            # Only spawn a certain number of objaverse objects per room, this number
            # differs for head and tail objects.
            if is_objaverse and (
                (
                    is_head
                    and total_head_objaverse_objects_placed
                    >= MAX_HEAD_OBJAVERSE_OBJECT_TYPES_PER_ROOM
                )
                or (
                    (not is_head)
                    and total_tail_objaverse_objects_placed
                    >= MAX_TAIL_OBJAVERSE_OBJECT_TYPES_PER_ROOM
                )
            ):
                del object_type_to_receptacle_infos[object_type_to_spawn]
                continue

            multiple_per_room = pt_db.PLACEMENT_ANNOTATIONS.loc[object_type_to_spawn][
                "multiplePerRoom"
            ]

            if (
                multiple_per_room
                and is_objaverse
                and is_head
                and not have_placed_multi_objaverse_head_cat
            ):
                # Here we try to place a head category object category multiple times if we haven't already
                multiplicity = max(
                    ri["multiplicity"]
                    for ri in object_type_to_receptacle_infos[object_type_to_spawn]
                )
                if multiplicity == 1:
                    multiplicity = min(
                        2, len(object_type_to_receptacle_infos[object_type_to_spawn])
                    )
            else:
                multiplicity = random.choice(
                    object_type_to_receptacle_infos[object_type_to_spawn]
                )["multiplicity"]

            assert multiplicity > 0, "Multiplicity must be greater than 0!"

            for _ in range(multiplicity):
                if (
                    object_type_to_count_in_room.get(object_type_to_spawn, 0) > 0
                    and not multiple_per_room
                ):
                    del object_type_to_receptacle_infos[object_type_to_spawn]
                    break

                # Here we intentionally randomly sample the spawn receptacle despite using the
                # multiplicity value from potentially a different spawn receptacle info. This is
                # to encourage spawning the object in different receptacles.
                receptacle_info = random.choice(
                    object_type_to_receptacle_infos[object_type_to_spawn]
                )

                receptacle_info["multiplicity"] -= 1
                if receptacle_info["multiplicity"] <= 0:
                    object_type_to_receptacle_infos[object_type_to_spawn].remove(
                        receptacle_info
                    )
                    if len(object_type_to_receptacle_infos[object_type_to_spawn]) == 0:
                        del object_type_to_receptacle_infos[object_type_to_spawn]

                split_sets = []
                for split_choice in split_to_split_choices(split=split):
                    if split == "train":
                        split_sets.append(["train", None])
                    else:
                        split_sets.append([split_choice])

                random.shuffle(split_sets)

                asset_candidates = None
                for split_set in split_sets:
                    asset_candidates = pt_db.ASSETS_DF[
                        (pt_db.ASSETS_DF["objectType"] == object_type_to_spawn)
                        & pt_db.ASSETS_DF["split"].isin(split_set)
                    ]  # TODO: This can be optimized

                    if object_type_to_spawn == "HousePlant":
                        # NOTE: House plants are a weird exception where there are massive
                        # house plants meant to be placed on the floor, and smaller
                        # house plants that can be placed on receptacles. This filters
                        # to only place smaller house plants on receptacles.
                        asset_candidates = asset_candidates[
                            asset_candidates["ySize"] < HOUSE_PLANT_MAX_HEIGHT
                        ]

                    if asset_candidates.shape[0] > 0:
                        break

                if asset_candidates.shape[0] == 0:
                    # NOTE: This can happen if we have a split that doesn't have any
                    # assets of this type, e.g. if we only have train assets for this type
                    # and we're trying to place a val or test object.
                    if object_type_to_spawn in object_type_to_receptacle_infos:
                        del object_type_to_receptacle_infos[object_type_to_spawn]
                    break

                # NOTE: Some objects have multiple sim object receptacles within it,
                # so we need to specify all of them as possible receptacle object ids.
                event = controller.step(action="ResetObjectFilter")
                receptacle_object_ids = [
                    obj["objectId"]
                    for obj in event.metadata["objects"]
                    if obj["objectId"].startswith(receptacle_info["receptacleId"])
                ]

                # TODO: Do we want to ensure these are always different?
                chosen_asset_id = asset_candidates.sample()["assetId"].iloc[0]

                generated_object_id = (
                    f"{object_type_to_spawn}|{room_id}|{total_small_objects_placed}"
                )

                # NOTE: spawn below the floor so it doesn't tip over any other objects.
                event = controller.step(
                    action="SpawnAsset",
                    assetId=chosen_asset_id,
                    generatedId=generated_object_id,
                    position=Vector3(x=0, y=FLOOR_Y - 20, z=0),
                    renderImage=False,
                )
                if not event:
                    warnings.warn(
                        f"{chosen_asset_id} failed to spawn (skipping), error message:\n{event.metadata['errorMessage']}"
                    )
                    continue

                # assert (
                #     event
                # ), f"SpawnAsset failed for {chosen_asset_id} with {event.metadata['errorMessage']}!"
                controller.step(
                    action="SetObjectFilter", objectIds=[generated_object_id]
                )

                openness = None
                if (
                    object_type_to_spawn in OPENNESS_RANDOMIZATIONS
                    and "CanOpen"
                    in pt_db.ASSET_ID_DATABASE[chosen_asset_id]["secondaryProperties"]
                ):
                    openness = sample_openness(object_type_to_spawn)
                    controller.step(
                        action="OpenObject",
                        objectId=generated_object_id,
                        openness=openness,
                        forceAction=True,
                        raise_for_failure=True,
                        renderImage=False,
                    )
                event = controller.step(
                    action="InitialRandomSpawn",
                    randomSeed=random.randint(0, 1_000_000_000),
                    objectIds=[generated_object_id],
                    receptacleObjectIds=receptacle_object_ids,
                    forceVisible=False,
                    allowFloor=False,
                    renderImage=False,
                    allowMoveable=True,
                )
                obj = next(
                    obj
                    for obj in event.metadata["objects"]
                    if obj["objectId"] == generated_object_id
                )
                center_position = obj["axisAlignedBoundingBox"]["center"].copy()

                # NOTE: Sometimes InitialRandomSpawn succeeds when it should
                # be failing. In these cases, the object will appear below
                # the floor.
                if not (event and center_position["y"] > FLOOR_Y):
                    controller.step(
                        action="DisableObject",
                        objectId=generated_object_id,
                        renderImage=False,
                    )
                    continue

                if obj["breakable"]:
                    # NOTE: often avoids objects shattering upon initialization.
                    center_position["y"] += 0.05

                states = {}
                if openness is not None:
                    states["openness"] = openness

                # NOTE: "___" is when there is a child SimObjPhysics of another
                # SimObjPhysics object (e.g., drawers on dressers).
                house_data_receptacle = receptacle_info["receptacleId"]
                if "___" in receptacle_info["receptacleId"]:
                    house_data_receptacle = receptacle_info["receptacleId"][
                        : receptacle_info["receptacleId"].find("___")
                    ]
                if "children" not in objects_in_house[house_data_receptacle]:
                    objects_in_house[house_data_receptacle]["children"] = []

                objects_in_house[house_data_receptacle]["children"].append(
                    Object(
                        id=generated_object_id,
                        assetId=chosen_asset_id,
                        objectType=object_type_to_spawn,
                        rotation=obj["rotation"],
                        position=center_position,
                        kinematic=bool(
                            pt_db.PLACEMENT_ANNOTATIONS.loc[object_type_to_spawn][
                                "isKinematic"
                            ]
                        ),
                        **states,
                    )
                )

                total_small_objects_placed += 1
                objects_types_placed_in_room.add(object_type_to_spawn)
                if object_type_to_spawn not in object_type_to_count_in_room:
                    object_type_to_count_in_room[object_type_to_spawn] = 0
                object_type_to_count_in_room[object_type_to_spawn] += 1

                if (
                    is_objaverse
                    and is_head
                    and object_type_to_count_in_room[object_type_to_spawn] > 1
                ):
                    have_placed_multi_objaverse_head_cat = True

                total_tail_objaverse_objects_placed += is_objaverse and (not is_head)
                total_head_objaverse_objects_placed += is_objaverse and is_head

    changed_ids = set()
    orig_objects = copy.deepcopy(partial_house.objects)
    _set_drop_heights(
        objects=partial_house.objects,
        changed_ids=changed_ids,
        object_type_to_prob_of_drop_change=FLOOR_OBJECTS_TO_DROP,
        object_type_to_prob_of_drop_change_children=OBJECTS_TO_DROP,
    )
    if changed_ids:
        controller.reset()
        event = controller.step(
            action="CreateHouse", house=partial_house.to_house_dict(), renderImage=False
        )
        assert event, "Unable to CreateHouse!"

        # NOTE: wait for objects to settle.
        last_objs = [
            obj for obj in event.metadata["objects"] if obj["objectId"] in changed_ids
        ]
        i = 0
        failed = False
        while True:
            i += 1
            if i > 1000:
                failed = True
                print("Objects not settling!")
                break

            event = controller.step(
                action="AdvancePhysicsStep",
                timeStep=0.01,
                allowAutoSimulation=True,
                renderImage=False,
            )
            objs = [
                obj
                for obj in event.metadata["objects"]
                if obj["objectId"] in changed_ids
            ]

            if all(
                all(
                    abs(obj["position"][k] - last_obj["position"][k]) < 1e-3
                    and abs(obj["rotation"][k] - last_obj["rotation"][k]) < 1e-3
                    for k in ["x", "y", "z"]
                )
                for obj, last_obj in zip(objs, last_objs)
            ):
                break
            last_objs = objs

        if failed:
            partial_house.objects = orig_objects
        else:
            _save_new_heights(
                changed_ids=changed_ids,
                partial_house_objects=partial_house.objects,
                thor_objects=event.metadata["objects"],
            )
