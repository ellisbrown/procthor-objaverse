from typing import Dict, Any

import prior

from ai2thor.controller import Controller


def get_objaverse_annotations():
    return prior.load_dataset(
        "objaverse-plus", revision="bce68ddc9f9dfbf1566d61dc4f04ac60e2f2d125"
    )["train"].data


class UpdateTHORMetadataWithObjaverseAnnotations:
    def __init__(self):
        self.objaverse_annotations = get_objaverse_annotations()

    def __call__(self, metadata: Dict[str, Any], controller: Controller):
        for o in metadata["objects"]:
            if o["assetId"] in self.objaverse_annotations:
                info = self.objaverse_annotations[o["assetId"]]
                o["objectType"] = o["objectId"].split("|")[0]
                o["description"] = info["description"]
