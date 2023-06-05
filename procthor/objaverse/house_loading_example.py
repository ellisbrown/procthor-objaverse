from sys import platform as sys_pf
from typing import Optional

import cv2
import matplotlib
import numpy as np
import pandas as pd
import prior

from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from procthor.constants import (
    PROCESSED_ASSET_DIRECTORY,
    PROCTHOR_INITIALIZATION,
)

matplotlib.use("MACOSX" if sys_pf == "darwin" else "TkAgg")

prompt_character_map = {
    "w": "MoveAhead",
    "s": "MoveBack",
    "a": "RotateLeft",
    "d": "RotateRight",
    "q": "End",
    "v": "Visualize",
    "1": "LookUp",
    "2": "LookDown",
}


def read_key_input(str):
    if str in prompt_character_map:
        return prompt_character_map[str]
    else:
        return None


def display_frame(
    frame, wait=None, plot_frame: Optional[np.ndarray] = None, height=480
):
    def rgb_to_bgr(f):
        return np.stack([f[:, :, 2], f[:, :, 1], f[:, :, 0]], axis=-1).astype(np.uint8)

    frame = rgb_to_bgr(frame)
    frame = cv2.resize(frame, (int(height * frame.shape[1] / frame.shape[0]), height))

    if plot_frame is not None:
        plot_frame = cv2.resize(
            plot_frame,
            (int(height * plot_frame.shape[1] / plot_frame.shape[0]), height),
        )
        frame = np.concatenate([frame, rgb_to_bgr(plot_frame)], axis=1)

    cv2.imshow("sample", frame)
    cv2.setWindowProperty("sample", cv2.WND_PROP_TOPMOST, 1)
    if wait is not None:
        return cv2.waitKey(wait)

    command = None
    while command is None:
        out = cv2.waitKey(wait)
        command = read_key_input(chr(out & 0xFF))

    return command


if __name__ == "__main__":
    # house = compress_json.load(
    # os.path.join(
    #     ABS_PATH_OF_TOP_LEVEL_PROCTHOR_DIR,
    #     "datasets/debug/0.json.gz",
    # )
    # )
    split = "test"
    dataset = prior.load_dataset(
        "procthor-objaverse-internal",
        revision="local",
        path_to_splits="/Users/lucaw/tmp/houses_v0.0.1",  # TODO: Change this to your local path
        max_houses_per_split=100,
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

    for ind, house in enumerate(dataset[split]):
        print(
            f"\nInteractive session in house {split}_{ind}, may take several seconds to load...\n"
        )

        c.reset(house)

        c.step("Teleport", **house["metadata"]["agent"])

        # print("Assets in house:")
        # print(set([o["assetId"] for o in c.last_event.metadata["objects"]]))

        while True:
            print("\nVisible objects:")
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.max_colwidth",
                10000,
                "display.expand_frame_repr",
                False,
            ):
                print(
                    pd.DataFrame(
                        [
                            {k: o[k] for k in ["objectType", "objectId", "assetId"]}
                            for o in c.last_event.metadata["objects"]
                            if o["visible"]
                        ]
                    )
                )

            action = display_frame(c.last_event.frame)

            if action == "End":
                break
            c.step(action)

    # def ma():
    #     c.step("MoveAhead")
    #     c.step("Pass")
    #
    #
    # def ml():
    #     c.step("MoveLeft")
    #     c.step("Pass")
    #
    #
    # def mr():
    #     c.step("MoveRight")
    #     c.step("Pass")
    #
    #
    # def lu():
    #     c.step("LookUp")
    #     c.step("Pass")
    #
    #
    # def ld():
    #     c.step("LookDown")
    #     c.step("Pass")
    #
    #
    # def rr():
    #     c.step("RotateRight")
    #     c.step("Pass")
    #
    #
    # def rl():
    #     c.step("RotateLeft")
    #     c.step("Pass")
    #
    #
    # def mb():
    #     c.step("MoveBack")
    #     c.step("Pass")
    #
    #
    # def cr():
    #     c.step("Crouch")
    #     c.step("Pass")
    #
    #
    # def st():
    #     c.step("Stand")
    #     c.step("Pass")
    #
    #
    # def vo():
    #     return [
    #         (o["objectType"], o["objectId"])
    #         for o in c.last_event.metadata["objects"]
    #         if o["visible"]
    #     ]
    #
    #
    # def va():
    #     return [
    #         (o["objectType"], o["assetId"])
    #         for o in c.last_event.metadata["objects"]
    #         if o["visible"]
    #     ]
    #
    #
    # def p():
    #     c.step("Pass")
    #
    #
    # print("Now debug")
    # print("Now debug")
    # print("Now debug")
    # print("Now debug")

    # counter = Counter(o["objectId"].split("|")[0] for o in c.last_event.metadata["objects"])
    #
    # oids = [
    #     o["objectId"]
    #     for o in c.last_event.metadata["objects"]
    #     if o["objectId"].startswith("ObjaHamburger")
    # ]
    #
    # oid0 = oids[0]
    # c.step(
    #     "Teleport",
    #     **c.step("GetInteractablePoses", objectId=oid0).metadata["actionReturn"][0]
    # )
    #
    # oid1 = oids[1]
    # c.step(
    #     "Teleport",
    #     **c.step("GetInteractablePoses", objectId=oid1).metadata["actionReturn"][0]
    # )
