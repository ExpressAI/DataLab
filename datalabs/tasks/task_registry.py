from typing import Dict

from datalabs.tasks import TaskTemplate, TaskType

TASK_REGISTRY: Dict = {}


def register_task(task: TaskType):
    def register_task_fn(cls):
        TASK_REGISTRY[task] = cls
        return cls

    return register_task_fn


def get_task(task: TaskType) -> TaskTemplate:
    return TASK_REGISTRY[task]
