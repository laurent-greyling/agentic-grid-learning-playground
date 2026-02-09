import random

def create_flag_location_with_hotspot_bias(
    grid_width_in_blocks: int,
    grid_height_in_blocks: int,
    hotspot_center_x_in_blocks: int,
    hotspot_center_y_in_blocks: int,
    hotspot_standard_deviation_in_blocks: float,
    hotspot_probability: float
) -> tuple[int, int]:
    """
    Returns a (flag_x_in_blocks, flag_y_in_blocks).

    With probability 'hotspot_probability', the flag is placed near a hotspot center (learnable bias).
    Otherwise, it is placed uniformly across the grid (noise).
    """

    random_value_between_0_and_1 = random.random()

    if random_value_between_0_and_1 < hotspot_probability:
        sampled_x = int(random.gauss(hotspot_center_x_in_blocks, hotspot_standard_deviation_in_blocks))
        sampled_y = int(random.gauss(hotspot_center_y_in_blocks, hotspot_standard_deviation_in_blocks))

        flag_x_in_blocks = max(0, min(grid_width_in_blocks - 1, sampled_x))
        flag_y_in_blocks = max(0, min(grid_height_in_blocks - 1, sampled_y))
        return flag_x_in_blocks, flag_y_in_blocks

    flag_x_in_blocks = random.randrange(grid_width_in_blocks)
    flag_y_in_blocks = random.randrange(grid_height_in_blocks)
    return flag_x_in_blocks, flag_y_in_blocks
