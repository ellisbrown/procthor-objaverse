import random
from collections import defaultdict
from typing import Dict, List, Union

from attr import field
from attrs import define

from procthor.generation.room_specs import RoomSpec, RoomSpecSampler
from procthor.utils.types import LeafRoom, MetaRoom


@define
class RoomSpecSamplerArgs:
    num_rooms: int = 1
    k: int = 1


DEFAULT_ROOM_SPEC_SAMPLER_ARGS = RoomSpecSamplerArgs()

ROOM_SPEC_ID_BY_NUM_ROOMS = {
    1: ["living-room", "kitchen", "bathroom", "bedroom"],
    2: ["kitchen-living-room", "bedroom-bathroom"],
    3: ["kitchen-living-bedroom-room", "kitchen-living-bedroom-room2"],
    4: ["4-room"],
    5: ["2-bed-1-bath", "5-room"],
    6: ["2-bed-2-bath"],
    7: ["7-room-3-bed"],
    8: ["8-room-3-bed"],
    # 10: ["12-room", "12-room-3-bed"],
}


@define
class LayoutSizeRoomSpecSampler(RoomSpecSampler):

    room_spec_by_num_rooms: Dict[int, List[RoomSpec]] = field(init=False)
    weights_all: List[int] = []

    def get_room_spec(self, room_spec_id):
        for room_spec in self.room_specs:
            if room_spec.room_spec_id == room_spec_id:
                return room_spec

    def __attrs_post_init__(self) -> None:
        self.room_spec_by_num_rooms = defaultdict(list)
        self.weights = defaultdict(list)
        self.weights_all = []
        for num_rooms, room_spec_ids in ROOM_SPEC_ID_BY_NUM_ROOMS.items():
            for room_spec_id in room_spec_ids:
                room_spec = self.get_room_spec(room_spec_id)
                self.room_spec_by_num_rooms[num_rooms].append(room_spec)
                self.weights[num_rooms].append(room_spec.sampling_weight)

    def sample(
        self, args: RoomSpecSamplerArgs = DEFAULT_ROOM_SPEC_SAMPLER_ARGS
    ) -> Union[RoomSpec, List[RoomSpec]]:
        """Return a RoomSpec with weighted sampling."""
        # room_specs, weights = self.get_room_specs(args.num_rooms)
        room_specs = self.room_spec_by_num_rooms[args.num_rooms]
        weights = self.weights[args.num_rooms]
        sample = random.choices(room_specs, weights=weights, k=args.k)
        return sample[0] if args.k == 1 else sample


@define
class UniformRoomSpecSampler(RoomSpecSampler):

    uniform_rooms_specs: List[RoomSpec] = []
    room_spec_by_num_rooms: Dict[int, List[RoomSpec]] = field(init=False)
    weights_all: List[int] = []

    def get_room_spec(self, room_spec_id):
        for room_spec in self.room_specs:
            if room_spec.room_spec_id == room_spec_id:
                return room_spec

    def __attrs_post_init__(self) -> None:
        self.uniform_rooms_specs = []
        self.weights = []
        max_num_houses_per_size = 4
        for num_rooms, room_spec_ids in ROOM_SPEC_ID_BY_NUM_ROOMS.items():
            num_houses = len(room_spec_ids)
            new_room_spec_ids = room_spec_ids.copy()
            if num_houses < max_num_houses_per_size:
                new_room_spec_ids.extend(
                    [
                        random.choice(room_spec_ids)
                        for _ in range(max_num_houses_per_size - num_houses)
                    ]
                )
            # print(new_room_spec_ids)
            room_specs = [
                self.get_room_spec(room_spec_id) for room_spec_id in new_room_spec_ids
            ]
            room_spec_weights = [room_spec.sampling_weight for room_spec in room_specs]
            self.uniform_rooms_specs.extend(room_specs)
            self.weights.extend(room_spec_weights)

    def sample(
        self, args: RoomSpecSamplerArgs = DEFAULT_ROOM_SPEC_SAMPLER_ARGS
    ) -> Union[RoomSpec, List[RoomSpec]]:
        """Return a RoomSpec with weighted sampling."""
        # room_specs, weights = self.get_room_specs(args.num_rooms)
        weights = self.weights
        sample = random.choices(self.uniform_rooms_specs, weights=weights, k=args.k)
        return sample[0] if args.k == 1 else sample


ROOM_SPECS = [
    RoomSpec(
        dims=lambda: (random.randint(13, 16), random.randint(5, 8)),
        room_spec_id="8-room-3-bed",
        sampling_weight=1,
        spec=[
            MetaRoom(
                ratio=4,
                children=[
                    MetaRoom(
                        ratio=2,
                        children=[
                            LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
                            LeafRoom(room_id=3, ratio=3, room_type="LivingRoom"),
                        ],
                    ),
                    MetaRoom(
                        ratio=1,
                        children=[
                            LeafRoom(room_id=4, ratio=2, room_type="LivingRoom"),
                            LeafRoom(
                                room_id=5,
                                ratio=1,
                                room_type="Bathroom",
                                avoid_doors_from_metarooms=True,
                            ),
                        ],
                    ),
                ],
            ),
            MetaRoom(
                ratio=1,
                children=[
                    LeafRoom(room_id=6, ratio=1, room_type="Bedroom"),
                ],
            ),
            MetaRoom(
                ratio=1,
                children=[
                    LeafRoom(room_id=7, ratio=1, room_type="Bedroom"),
                ],
            ),
            MetaRoom(
                ratio=2,
                children=[
                    LeafRoom(room_id=8, ratio=1, room_type="Bedroom"),
                    LeafRoom(
                        room_id=9,
                        ratio=1,
                        room_type="Bathroom",
                        avoid_doors_from_metarooms=True,
                    ),
                ],
            ),
        ],
    ),
    RoomSpec(
        room_spec_id="7-room-3-bed",
        sampling_weight=1,
        spec=[
            MetaRoom(
                ratio=3,
                children=[
                    MetaRoom(
                        ratio=2,
                        children=[
                            LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
                            LeafRoom(room_id=3, ratio=3, room_type="LivingRoom"),
                        ],
                    ),
                    MetaRoom(
                        ratio=1,
                        children=[
                            LeafRoom(room_id=4, ratio=2, room_type="LivingRoom"),
                            LeafRoom(
                                room_id=5,
                                ratio=1,
                                room_type="Bathroom",
                                avoid_doors_from_metarooms=True,
                            ),
                        ],
                    ),
                ],
            ),
            MetaRoom(
                ratio=2,
                children=[
                    LeafRoom(room_id=6, ratio=2, room_type="Bedroom"),
                    LeafRoom(room_id=7, ratio=2, room_type="Bedroom"),
                    LeafRoom(room_id=8, ratio=2, room_type="Bedroom"),
                ],
            ),
        ],
    ),
    RoomSpec(
        room_spec_id="12-room-3-bed",
        sampling_weight=1,
        spec=[
            MetaRoom(
                ratio=1,
                children=[
                    MetaRoom(
                        ratio=1,
                        children=[
                            LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
                            LeafRoom(room_id=3, ratio=3, room_type="LivingRoom"),
                        ],
                    ),
                    MetaRoom(
                        ratio=1,
                        children=[
                            LeafRoom(room_id=4, ratio=1, room_type="LivingRoom"),
                            LeafRoom(room_id=5, ratio=1, room_type="LivingRoom"),
                        ],
                    ),
                ],
            ),
            MetaRoom(
                ratio=1,
                children=[
                    MetaRoom(
                        ratio=1,
                        children=[
                            LeafRoom(room_id=6, ratio=2, room_type="Bedroom"),
                            LeafRoom(
                                room_id=7,
                                ratio=1,
                                room_type="Bathroom",
                                avoid_doors_from_metarooms=True,
                            ),
                        ],
                    ),
                    MetaRoom(
                        ratio=1,
                        children=[
                            LeafRoom(room_id=8, ratio=2, room_type="Bedroom"),
                            LeafRoom(
                                room_id=9,
                                ratio=1,
                                room_type="Bathroom",
                                avoid_doors_from_metarooms=True,
                            ),
                        ],
                    ),
                    MetaRoom(
                        ratio=1,
                        children=[
                            LeafRoom(room_id=10, ratio=2, room_type="Bedroom"),
                            LeafRoom(
                                room_id=11,
                                ratio=1,
                                room_type="Bathroom",
                                avoid_doors_from_metarooms=True,
                            ),
                        ],
                    ),
                ],
            ),
        ],
    ),
    RoomSpec(
        room_spec_id="12-room",
        sampling_weight=1,
        spec=[
            MetaRoom(
                ratio=2,
                children=[
                    MetaRoom(
                        ratio=1,
                        children=[
                            LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
                            LeafRoom(room_id=3, ratio=3, room_type="LivingRoom"),
                        ],
                    ),
                    MetaRoom(
                        ratio=1,
                        children=[
                            LeafRoom(room_id=4, ratio=2, room_type="Bedroom"),
                            LeafRoom(
                                room_id=5,
                                ratio=1,
                                room_type="Bathroom",
                                avoid_doors_from_metarooms=True,
                            ),
                        ],
                    ),
                ],
            ),
            MetaRoom(
                ratio=3,
                children=[
                    MetaRoom(
                        ratio=1,
                        children=[
                            LeafRoom(room_id=6, ratio=2, room_type="Bedroom"),
                            LeafRoom(
                                room_id=7,
                                ratio=1,
                                room_type="Bathroom",
                                avoid_doors_from_metarooms=True,
                            ),
                        ],
                    ),
                    MetaRoom(
                        ratio=1,
                        children=[
                            LeafRoom(room_id=8, ratio=2, room_type="Bedroom"),
                            LeafRoom(
                                room_id=9,
                                ratio=1,
                                room_type="Bathroom",
                                avoid_doors_from_metarooms=True,
                            ),
                        ],
                    ),
                    MetaRoom(
                        ratio=1,
                        children=[
                            LeafRoom(room_id=10, ratio=2, room_type="Bedroom"),
                            LeafRoom(
                                room_id=11,
                                ratio=1,
                                room_type="Bathroom",
                                avoid_doors_from_metarooms=True,
                            ),
                        ],
                    ),
                ],
            ),
        ],
    ),
    RoomSpec(
        room_spec_id="4-room",
        sampling_weight=1,
        spec=[
            MetaRoom(
                ratio=2,
                children=[
                    LeafRoom(room_id=4, ratio=2, room_type="Bedroom"),
                    LeafRoom(
                        room_id=5,
                        ratio=1,
                        room_type="Bathroom",
                        avoid_doors_from_metarooms=True,
                    ),
                ],
            ),
            MetaRoom(
                ratio=2,
                children=[
                    LeafRoom(room_id=6, ratio=3, room_type="Kitchen"),
                    LeafRoom(room_id=7, ratio=2, room_type="LivingRoom"),
                ],
            ),
        ],
    ),
    RoomSpec(
        room_spec_id="2-bed-1-bath",
        sampling_weight=1,
        spec=[
            MetaRoom(
                ratio=2,
                children=[
                    LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
                    LeafRoom(
                        room_id=3,
                        ratio=2,
                        room_type="Bathroom",
                        avoid_doors_from_metarooms=True,
                    ),
                    LeafRoom(room_id=4, ratio=3, room_type="LivingRoom"),
                ],
            ),
            LeafRoom(room_id=5, ratio=1, room_type="Bedroom"),
            LeafRoom(room_id=6, ratio=1, room_type="Bedroom"),
        ],
    ),
    RoomSpec(
        room_spec_id="5-room",
        sampling_weight=1,
        spec=[
            MetaRoom(
                ratio=2,
                children=[
                    LeafRoom(room_id=4, ratio=2, room_type="Bedroom"),
                    LeafRoom(
                        room_id=5,
                        ratio=1,
                        room_type="Bathroom",
                        avoid_doors_from_metarooms=True,
                    ),
                ],
            ),
            LeafRoom(room_id=6, ratio=2, room_type="Bedroom"),
            MetaRoom(
                ratio=2,
                children=[
                    LeafRoom(room_id=8, ratio=3, room_type="Kitchen"),
                    LeafRoom(room_id=9, ratio=2, room_type="LivingRoom"),
                ],
            ),
        ],
    ),
    RoomSpec(
        room_spec_id="2-bed-2-bath",
        sampling_weight=1,
        spec=[
            MetaRoom(
                ratio=2,
                children=[
                    LeafRoom(room_id=4, ratio=2, room_type="Bedroom"),
                    LeafRoom(
                        room_id=5,
                        ratio=1,
                        room_type="Bathroom",
                        avoid_doors_from_metarooms=True,
                    ),
                ],
            ),
            MetaRoom(
                ratio=2,
                children=[
                    LeafRoom(room_id=6, ratio=2, room_type="Bedroom"),
                    LeafRoom(
                        room_id=7,
                        ratio=1,
                        room_type="Bathroom",
                        avoid_doors_from_metarooms=True,
                    ),
                ],
            ),
            MetaRoom(
                ratio=2,
                children=[
                    LeafRoom(room_id=8, ratio=3, room_type="Kitchen"),
                    LeafRoom(room_id=9, ratio=2, room_type="LivingRoom"),
                ],
            ),
        ],
    ),
    RoomSpec(
        room_spec_id="bedroom-bathroom",
        sampling_weight=1,
        spec=[
            LeafRoom(room_id=2, ratio=2, room_type="Bedroom"),
            LeafRoom(room_id=3, ratio=1, room_type="Bathroom"),
        ],
    ),
    RoomSpec(
        room_spec_id="kitchen-living-bedroom-room",
        sampling_weight=1,
        spec=[
            MetaRoom(
                ratio=2,
                children=[
                    LeafRoom(room_id=6, ratio=3, room_type="Kitchen"),
                    LeafRoom(room_id=7, ratio=2, room_type="LivingRoom"),
                ],
            ),
            LeafRoom(room_id=2, ratio=1, room_type="Bedroom"),
        ],
    ),
    RoomSpec(
        room_spec_id="kitchen-living-bedroom-room2",
        sampling_weight=1,
        spec=[
            MetaRoom(
                ratio=2,
                children=[
                    LeafRoom(room_id=6, ratio=1, room_type="Kitchen"),
                    LeafRoom(room_id=7, ratio=1, room_type="LivingRoom"),
                ],
            ),
            LeafRoom(room_id=2, ratio=1, room_type="Bedroom"),
        ],
    ),
    RoomSpec(
        room_spec_id="kitchen-living-room",
        sampling_weight=1,
        spec=[
            LeafRoom(room_id=2, ratio=1, room_type="Kitchen"),
            LeafRoom(room_id=3, ratio=1, room_type="LivingRoom"),
        ],
    ),
    RoomSpec(
        room_spec_id="kitchen",
        sampling_weight=1,
        spec=[LeafRoom(room_id=2, ratio=1, room_type="Kitchen")],
    ),
    RoomSpec(
        room_spec_id="living-room",
        sampling_weight=1,
        spec=[LeafRoom(room_id=2, ratio=1, room_type="LivingRoom")],
    ),
    RoomSpec(
        room_spec_id="bedroom",
        sampling_weight=1,
        spec=[LeafRoom(room_id=2, ratio=1, room_type="Bedroom")],
    ),
    RoomSpec(
        # scale=1.25?
        room_spec_id="bathroom",
        sampling_weight=1,
        spec=[LeafRoom(room_id=2, ratio=1, room_type="Bathroom")],
    ),
]

LAYOUT_SIZE_ROOM_SPEC_SAMPLER = LayoutSizeRoomSpecSampler(room_specs=ROOM_SPECS)
