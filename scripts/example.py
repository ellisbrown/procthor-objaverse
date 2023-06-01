import random

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
    RoomSpecSampler,
    RoomSpec,
)
from procthor.objaverse.objaverse_add_object_functions import (
    objaverse_add_floor_objects,
    objaverse_add_wall_objects,
    objaverse_add_small_objects,
)
from procthor.objaverse.objaverse_constants import DEFAULT_OBJAVERSE_PROCTHOR_DATABASE
from procthor.utils.types import LeafRoom, SamplingVars


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
