"""
Generators for synthetic data creation.
"""

from .trajectory_generator import (
    TrajectoryGenerator,
    QuestionGenerator,
    Trajectory,
    ToolCall
)

from .trajectory_generator_v2 import (
    TrajectoryGeneratorV2
)

# Phase 2: Multi-Iteration Generator (NEW)
from .trajectory_generator_multi_iter import (
    TrajectoryGeneratorMultiIter,
    TrainingExample
)

__all__ = [
    # Phase 1 Generators
    'TrajectoryGenerator',
    'TrajectoryGeneratorV2',
    'QuestionGenerator',
    'Trajectory',
    'ToolCall',
    # Phase 2 Generators (NEW)
    'TrajectoryGeneratorMultiIter',
    'TrainingExample',
]




# """
# Generators for synthetic data creation.
# """

# from .trajectory_generator import (
#     TrajectoryGenerator,
#     QuestionGenerator,
#     Trajectory,
#     ToolCall
# )

# from .trajectory_generator_v2_backup import (
#     TrajectoryGeneratorV2
# )

# __all__ = [
#     'TrajectoryGenerator',
#     'TrajectoryGeneratorV2',
#     'QuestionGenerator',
#     'Trajectory',
#     'ToolCall'
# ]