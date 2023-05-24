import logging
import os
import platform
import queue
import sys
import time
import traceback
from datetime import datetime
from time import sleep
from typing import Sequence, List

import numpy as np
import torch

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from procthor.constants import PROCTHOR_INITIALIZATION
from procthor.generation import HouseGenerator
from procthor.objaverse.house_generation.room_spec_sampler import (
    ROOM_SPECS,
    UniformRoomSpecSampler,
)
import multiprocessing as mp

from procthor.objaverse.objaverse_constants import DEFAULT_OBJAVERSE_PROCTHOR_DATABASE
from scripts.example import _create_objaverse_generation_functions

mp = mp.get_context("spawn" if sys.platform == "darwin" else "forkserver")
# mp = mp.get_context("forkserver")


def partition_sequence(seq: Sequence, parts: int) -> List:
    assert 0 < parts, f"parts [{parts}] must be greater > 0"
    assert parts <= len(seq), f"parts [{parts}] > len(seq) [{len(seq)}]"
    n = len(seq)

    quotient = n // parts
    remainder = n % parts
    counts = [quotient + (i < remainder) for i in range(parts)]
    inds = np.cumsum([0] + counts)
    return [seq[ind0:ind1] for ind0, ind1 in zip(inds[:-1], inds[1:])]


def generate_house(worker_ind: int, split: str, in_queue: mp.Queue) -> None:
    print(f"Worker {worker_ind} start")

    device_kwargs = {}
    if sys.platform != "darwin":
        device_kwargs = dict(
            gpu_device=worker_ind % torch.cuda.device_count(),
            platform=CloudRendering,
        )

    def create_controller():
        return Controller(
            quality="Very Low",
            width=100,
            height=100,
            **device_kwargs,
            **PROCTHOR_INITIALIZATION,
        )

    room_spec_sampler = UniformRoomSpecSampler(room_specs=ROOM_SPECS)

    house_generator = HouseGenerator(
        controller=None,
        split=split,
        room_spec_sampler=room_spec_sampler,
        pt_db=DEFAULT_OBJAVERSE_PROCTHOR_DATABASE,
        generation_functions=_create_objaverse_generation_functions(),
    )

    save_dir = f"datasets/procthor-objaverse/{split}"
    os.makedirs(save_dir, exist_ok=True)

    houses_generated = 0
    controller = None
    try:
        while True:
            house_inds = in_queue.get(timeout=30)

            for house_ind in house_inds:
                save_path = os.path.join(save_dir, f"{house_ind}.json.gz")

                if os.path.exists(save_path):
                    print(f"Worker {worker_ind} skipping {house_ind}, already exists")
                    continue

                if houses_generated % 20 == 0:
                    if controller != None:
                        controller.stop()  # TODO: Unload assets somehow instead of doing this
                    controller = create_controller()
                    house_generator.controller = controller

                houses_generated += 1
                house_generator.set_seed(house_ind)

                # NOTE: sometimes house_generator.sample() hangs
                room_spec = None
                while True:
                    try:
                        house_generator.room_spec = room_spec
                        house, _ = house_generator.sample()
                        house.validate(house_generator.controller)
                        if house.data["metadata"]["warnings"]:
                            # NOTE: Keep the room spec the same to avoid sampling bias.
                            room_spec = house.room_spec
                            continue
                    except (AssertionError, KeyError):
                        house_generator.room_spec = room_spec
                        print(
                            f"Worker {worker_ind} encountered an exception for {house_ind},"
                            f" retrying... Exception:\n{traceback.format_exc()}"
                        )
                        continue
                    except Exception as e:
                        logging.error(traceback.format_exc())
                        house = None

                    break

                print(f"Worker {worker_ind} finished house {house_ind}")

                if house is not None:
                    house.to_json(
                        save_path,
                        compressed=True,
                    )
                else:
                    print(f"Worker {worker_ind}: skipping house {house_ind}")
    except queue.Empty:
        try:
            controller.stop()
        except:
            pass
        print(f"Worker {worker_ind} finished")


if __name__ == "__main__":
    on_server = sys.platform != "darwin"

    nprocesses = 60 if on_server else 1
    nhouses = 100_000 if on_server else 1000
    split = "train"
    in_queue = mp.Queue()

    for house_inds in partition_sequence(list(range(nhouses)), min(30000, nhouses)):
        in_queue.put(house_inds)

    processes = []
    for worker_ind in range(nprocesses):
        p = mp.Process(
            target=generate_house,
            kwargs=dict(worker_ind=worker_ind, split=split, in_queue=in_queue),
        )
        p.start()
        processes.append(p)

        time.sleep(1)

    for p in processes:
        p.join()
