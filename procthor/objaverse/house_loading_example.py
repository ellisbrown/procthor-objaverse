import os

import compress_json

from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from procthor.constants import (
    PROCESSED_ASSET_DIRECTORY,
    PROCTHOR_INITIALIZATION,
    ABS_PATH_OF_TOP_LEVEL_PROCTHOR_DIR,
)
from collections import Counter

if __name__ == "__main__":
    house = compress_json.load(
        os.path.join(
            ABS_PATH_OF_TOP_LEVEL_PROCTHOR_DIR,
            "datasets/procthor-objaverse/train/0.json.gz",
        )
    )

    c = Controller(
        scene="Procedural",
        commit_id=PROCTHOR_INITIALIZATION["commit_id"],
        height=1000,
        width=1000,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=PROCESSED_ASSET_DIRECTORY,
            asset_symlink=True,
            verbose=True,
        ),
    )

    c.reset(house)

    c.step("Teleport", **house["metadata"]["agent"])

    print(set([o["assetId"] for o in c.last_event.metadata["objects"]]))

    def ma():
        c.step("MoveAhead")
        c.step("Pass")

    def ml():
        c.step("MoveLeft")
        c.step("Pass")

    def mr():
        c.step("MoveRight")
        c.step("Pass")

    def lu():
        c.step("LookUp")
        c.step("Pass")

    def ld():
        c.step("LookDown")
        c.step("Pass")

    def rr():
        c.step("RotateRight")
        c.step("Pass")

    def rl():
        c.step("RotateLeft")
        c.step("Pass")

    def mb():
        c.step("MoveBack")
        c.step("Pass")

    def cr():
        c.step("Crouch")
        c.step("Pass")

    def st():
        c.step("Stand")
        c.step("Pass")

    def vo():
        return [
            (o["objectType"], o["objectId"])
            for o in c.last_event.metadata["objects"]
            if o["visible"]
        ]

    def va():
        return [
            (o["objectType"], o["assetId"])
            for o in c.last_event.metadata["objects"]
            if o["visible"]
        ]

    def p():
        c.step("Pass")

    print("Now debug")
    print("Now debug")
    print("Now debug")
    print("Now debug")

    counter = Counter(
        o["objectId"].split("|")[0] for o in c.last_event.metadata["objects"]
    )

    oids = [
        o["objectId"]
        for o in c.last_event.metadata["objects"]
        if o["objectId"].startswith("ObjaHamburger")
    ]

    oid0 = oids[0]
    c.step(
        "Teleport",
        **c.step("GetInteractablePoses", objectId=oid0).metadata["actionReturn"][0]
    )

    oid1 = oids[1]
    c.step(
        "Teleport",
        **c.step("GetInteractablePoses", objectId=oid1).metadata["actionReturn"][0]
    )
