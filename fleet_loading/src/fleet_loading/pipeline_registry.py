from kedro.pipeline import Pipeline

from .pipelines.training.pipeline import create_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    training = create_pipeline()
    return {
        "__default__": training,
        "training": training,
    }
