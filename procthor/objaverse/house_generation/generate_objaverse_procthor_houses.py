import argparse
import logging
import multiprocessing as mp
import os
import queue
import string
import sys
import time
import traceback
from typing import Sequence, List, Optional

import numpy as np
import torch

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from procthor.constants import (
    PROCTHOR_INITIALIZATION,
)
from procthor.generation import (
    HouseGenerator,
    GenerationFunctions,
    default_sample_house_structure,
    default_add_doors,
    default_add_lights,
    default_add_skybox,
    default_add_exterior_walls,
    default_add_rooms,
    default_randomize_object_colors,
    default_randomize_object_states,
)
from procthor.generation.house import simple_benchmark_thor
from procthor.objaverse.house_generation.room_spec_sampler import (
    ROOM_SPECS,
    UniformRoomSpecSampler,
)
from procthor.objaverse.objaverse_add_object_functions import (
    objaverse_add_floor_objects,
    objaverse_add_wall_objects,
    objaverse_add_small_objects,
)
from procthor.objaverse.objaverse_databases import DEFAULT_OBJAVERSE_PROCTHOR_DATABASE
from procthor.utils.types import Split

mp = mp.get_context("spawn" if sys.platform == "darwin" else "forkserver")


def _create_objaverse_generation_functions():
    return GenerationFunctions(
        sample_house_structure=default_sample_house_structure,
        add_doors=default_add_doors,
        add_lights=default_add_lights,
        add_skybox=default_add_skybox,
        add_exterior_walls=default_add_exterior_walls,
        add_rooms=default_add_rooms,
        add_floor_objects=objaverse_add_floor_objects,
        add_wall_objects=objaverse_add_wall_objects,
        add_small_objects=objaverse_add_small_objects,
        randomize_object_colors=default_randomize_object_colors,
        randomize_object_states=default_randomize_object_states,
    )


def partition_sequence(seq: Sequence, parts: int) -> List:
    assert 0 < parts, f"parts [{parts}] must be greater > 0"
    assert parts <= len(seq), f"parts [{parts}] > len(seq) [{len(seq)}]"
    n = len(seq)

    quotient = n // parts
    remainder = n % parts
    counts = [quotient + (i < remainder) for i in range(parts)]
    inds = np.cumsum([0] + counts)
    return [seq[ind0:ind1] for ind0, ind1 in zip(inds[:-1], inds[1:])]


def parse_queue_id(id: str, expected_split: str):
    assert set(id) <= set(
        string.ascii_letters + string.digits + "_"
    ), f"Invalid ID '{id}', id must only contain a-Z, 0-9, and _"

    # Parse the ID into split, house
    run_info = {k: v for part in id.split("__") for k, v in [part.split("_")]}

    split = run_info["split"]
    assert expected_split == split
    house_index = int(run_info["house"])

    return {
        "id": id,
        "split": split,
        "house_index": int(house_index),
    }


def generate_house(
    worker_ind: int, split: Split, in_queue: mp.Queue, save_dir: str
) -> None:
    print(f"Worker {worker_ind}: start")
    assert os.path.exists(
        os.path.dirname(save_dir)
    ), f"{os.path.dirname(save_dir)} must exist"

    device_kwargs = {}
    if sys.platform != "darwin":
        device_kwargs = dict(
            gpu_device=worker_ind % torch.cuda.device_count(),
            platform=CloudRendering,
        )

    def create_controller():
        assert os.path.exists(
            PROCTHOR_INITIALIZATION["action_hook_runner"].asset_directory
        )
        return Controller(
            quality="Very Low",
            width=100,
            height=100,
            **device_kwargs,
            **PROCTHOR_INITIALIZATION,
            server_timeout=600,
            server_start_timeout=600,
        )

    room_spec_sampler = UniformRoomSpecSampler(room_specs=ROOM_SPECS)

    house_generator = HouseGenerator(
        controller=None,
        split=split,
        room_spec_sampler=room_spec_sampler,
        pt_db=DEFAULT_OBJAVERSE_PROCTHOR_DATABASE,
        generation_functions=_create_objaverse_generation_functions(),
    )

    os.makedirs(save_dir, exist_ok=True)

    houses_generated = 0
    controller: Optional[Controller] = None
    min_fps: Optional[float] = None
    consecutive_failures = 0
    try:
        while True:
            ids = in_queue.get(timeout=30)

            if isinstance(ids, str):
                ids = [ids]
            else:
                assert isinstance(ids, list)

            house_inds = [
                parse_queue_id(id, expected_split=split)["house_index"] for id in ids
            ]

            for house_ind in house_inds:
                start_time = time.time()
                save_path = os.path.join(save_dir, f"{house_ind}.json.gz")

                if os.path.exists(save_path):
                    print(f"Worker {worker_ind}: skipping {house_ind}, already exists")
                    continue

                if houses_generated % 10 == 0 or controller is None:
                    if controller != None:
                        controller.stop()  # TODO: Unload assets somehow instead of doing this

                    controller = create_controller()
                    house_generator.controller = controller

                    controller.reset("FloorPlan1")
                    bench_fps = simple_benchmark_thor(controller)

                    min_fps = 0.25 * bench_fps  # Don't go below 25% of the iTHOR FPS

                    print(f"Worker {worker_ind}: min_fps = {min_fps})")
                    controller.reset("Procedural")

                houses_generated += 1
                house_generator.set_seed(house_ind)

                # NOTE: sometimes house_generator.sample() hangs
                room_spec = None
                while True:
                    if consecutive_failures == 20:
                        raise RuntimeError(
                            f"Worker {worker_ind}: failed 20 times in a row, exiting"
                        )
                    try:
                        house_generator.room_spec = room_spec
                        house, _ = house_generator.sample()
                        house.validate(house_generator.controller, min_fps=min_fps)
                        if house.data["metadata"]["warnings"]:
                            consecutive_failures += 1
                            # NOTE: Keep the room spec the same to avoid sampling bias.
                            room_spec = house.room_spec
                            continue

                        consecutive_failures = 0
                    except (AssertionError, KeyError):
                        house_generator.room_spec = room_spec
                        print(
                            f"Worker {worker_ind}: encountered an exception for {house_ind},"
                            f" retrying... Exception:\n{traceback.format_exc()}"
                        )
                        consecutive_failures += 1
                        continue
                    except Exception:
                        logging.error(traceback.format_exc())
                        house = None
                        consecutive_failures += 1

                    break

                time_taken = f"took {time.time() - start_time:.1f}s"
                if house is None:
                    print(
                        f"Worker {worker_ind}: finished house {house_ind} (FAILURE, {time_taken}) skipping house {house_ind}"
                    )
                else:
                    objects_in_json = []
                    for object in house.data["objects"]:
                        objects_in_json.append(object)
                        objects_in_json.extend(object.get("children", []))

                    objaverse_objs_from_controller = [
                        o
                        for o in controller.last_event.metadata["objects"]
                        if o["name"].startswith("Obja")
                    ]
                    print(
                        f"Worker {worker_ind}: finished house {house_ind} (SUCCESS, {time_taken})"
                        f" {len(house.rooms)} rooms,"
                        f" {len(objects_in_json)} objects,"
                        f" {len(objaverse_objs_from_controller)} objaverse objects"
                        f" from {len(set(o['name'].split('|')[0] for o in objaverse_objs_from_controller))} unique classes"
                        f" {len([o for o in objaverse_objs_from_controller if o['pickupable']])} pickupable"
                        f" {len([o for o in objaverse_objs_from_controller if ((not o['pickupable']) and o['moveable'])])} moveable"
                        f" {len([o for o in objaverse_objs_from_controller if not (o['pickupable'] or o['moveable'])])} neither"
                    )
                    house.to_json(
                        save_path,
                        compressed=True,
                    )
    except queue.Empty:
        pass
    finally:
        try:
            controller.stop()
        except:
            pass
        print(f"Worker {worker_ind}: finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating houses")
    parser.add_argument(
        "--save_dir",
        type=str,
        help="""The directory to save the dataset to.""",
        required=True,
    )
    parser.add_argument(
        "--split", type=str, help="""The train, val, or test split.""", required=True
    )
    args = parser.parse_args()

    on_server = sys.platform != "darwin"

    nprocesses = torch.cuda.device_count() * 15 if on_server else 1

    if args.split == "train":
        nhouses = 150_000
    elif args.split in ["val", "test"]:
        nhouses = 15_000
    else:
        raise NotImplementedError

    in_queue = mp.Queue()

    for house_inds in partition_sequence(list(range(nhouses)), min(30000, nhouses)):
        in_queue.put([f"split_{args.split}__house_{i}" for i in house_inds])

    processes = []
    for worker_ind in range(nprocesses):
        p = mp.Process(
            target=generate_house,
            kwargs=dict(
                worker_ind=worker_ind,
                split=args.split,
                save_dir=args.save_dir,
                in_queue=in_queue,
            ),
        )

        p.start()
        processes.append(p)

        time.sleep(1)

    for p in processes:
        p.join()
