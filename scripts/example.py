import random

from procthor.generation import (
    HouseGenerator,
    RoomSpecSampler,
    RoomSpec,
)
from procthor.objaverse.house_generation.generate_objaverse_procthor_houses import (
    _create_objaverse_generation_functions,
)
from procthor.objaverse.objaverse_databases import DEFAULT_OBJAVERSE_PROCTHOR_DATABASE
from procthor.utils.types import LeafRoom, SamplingVars

if __name__ == "__main__":
    room_spec_sampler = RoomSpecSampler(
        [
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
    )

    house_generator = HouseGenerator(
        split="train",
        seed=41,
        room_spec_sampler=room_spec_sampler,
        pt_db=DEFAULT_OBJAVERSE_PROCTHOR_DATABASE,
        generation_functions=_create_objaverse_generation_functions(),
    )
    sampling_vars = SamplingVars(
        interior_boundary_scale=random.uniform(1.6, 2.2),
        max_floor_objects=10,
    )
    house, _ = house_generator.sample()
    house.validate(house_generator.controller)

    house.to_json("temp.json")
