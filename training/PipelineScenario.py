from enum import Enum


class PipelineScenario(Enum):
    NATIVE_CUMULATIVE = 0

    NATIVE_CWR_STAR = 1
    NATIVE_AR1_STAR = 2
    NATIVE_AR1_STAR_FREE = 3

    LR_CWR_STAR = 4
    LR_AR1_STAR = 5
    LR_AR1_STAR_FREE = 6


PIPELINES_WITH_RM = [
    PipelineScenario.NATIVE_CWR_STAR,
    PipelineScenario.NATIVE_AR1_STAR,
    PipelineScenario.NATIVE_AR1_STAR_FREE,
    PipelineScenario.LR_CWR_STAR,
    PipelineScenario.LR_AR1_STAR,
    PipelineScenario.LR_AR1_STAR_FREE,
]

PIPELINES_WITH_LEARNING_RATE_MODULATION = [
    PipelineScenario.NATIVE_AR1_STAR,
    PipelineScenario.LR_AR1_STAR,
]

LR_PIPELINES = [
    PipelineScenario.LR_CWR_STAR,
    PipelineScenario.LR_AR1_STAR,
    PipelineScenario.LR_AR1_STAR_FREE,
]

CWR_STAR_PIPELINES = [
    PipelineScenario.NATIVE_CWR_STAR,
    PipelineScenario.LR_CWR_STAR,
]

AR1_STAR_PURE_PIPELINES = [
    PipelineScenario.NATIVE_AR1_STAR,
    PipelineScenario.LR_AR1_STAR,
]

AR1_STAR_FREE_PIPELINES = [
    PipelineScenario.NATIVE_AR1_STAR_FREE,
    PipelineScenario.LR_AR1_STAR_FREE,
]
