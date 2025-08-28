"""Task generators for spatial feature learning scenarios.

This module provides high-level interfaces for creating common
spatial learning tasks like:
- Binary spatial classification (which star is on the left?)
- Multi-object positioning tasks
- Size/color comparison tasks
- Complex spatial reasoning scenarios
"""

from tiled_dummy_gen.tasks.generators import (
    TaskGenerator,
    SpatialBinaryTaskGenerator,
    ColorComparisonTaskGenerator,
    SizeComparisonTaskGenerator,
    MultiObjectTaskGenerator,
)

from tiled_dummy_gen.tasks.presets import (
    create_left_right_task,
    create_above_below_task,
    create_color_identification_task,
    create_size_comparison_task,
    create_star_positioning_task,
)

__all__ = [
    "TaskGenerator",
    "SpatialBinaryTaskGenerator",
    "ColorComparisonTaskGenerator",
    "SizeComparisonTaskGenerator",
    "MultiObjectTaskGenerator",
    "create_left_right_task",
    "create_above_below_task",
    "create_color_identification_task",
    "create_size_comparison_task",
    "create_star_positioning_task",
]
