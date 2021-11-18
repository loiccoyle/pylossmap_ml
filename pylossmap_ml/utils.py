from typing import List

from .db import DB

PARTICLE_TYPE_VARS = ["LHC.STATS:B1_PARTICLE_TYPE", "LHC.STATS:B2_PARTICLE_TYPE"]
PARTICLE_TYPE_MAP = {"PB82": "ion", "PROTON": "proton"}


def get_fill_particle(fill_number: int) -> List[str]:
    """Get the particle type of the fill for both beam.

    Args:
        fill_number: the fill to check.


    """
    fill_info = DB.getLHCFillData(fill_number)
    timber_out = DB.get(PARTICLE_TYPE_VARS, fill_info["startTime"] + 60)
    return [
        PARTICLE_TYPE_MAP[beam_part_type[1][0]]
        for beam_part_type in timber_out.values()
    ]
