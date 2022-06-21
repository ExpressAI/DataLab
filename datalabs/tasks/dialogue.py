from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import ClassLabel, Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.dialogue)
@dataclass
class DialogueGeneration(TaskTemplate):
    task: TaskType = TaskType.dialogue
    dialogue_column: str = "dialog"

@register_task(TaskType.dialogue_emotion_action_tracking)
@dataclass
class DialogueEmotionActionTracking(DialogueGeneration):
    task: TaskType = TaskType.dialogue_emotion_action_tracking
    input_schema: ClassVar[Features] = Features(
        {
            "dialog": Sequence(Value("string")),
        }
    )
    label_schema: ClassVar[Features] = Features(
        {
            "act": Sequence(Value("int32")),
            "emotion": Sequence(Value("int32")),
        }
    )
    dialog_column: str = "dialog"
    act_column: str = "act"
    emotion_column: str = "emotion"


@register_task(TaskType.dialogue_empathetic)
@dataclass
class DialogueEmpathetic(DialogueGeneration):
    task: TaskType = TaskType.dialogue_empathetic
    input_schema: ClassVar[Features] = Features(
        {
            "situation": Value("string"),
            "utterance": Value("string"),
        }
    )
    label_schema: ClassVar[Features] = Features(
        {
            "emotion": Value("string"),
        }
    )
    situation_column: str = "situation"
    utterance_column: str = "utterance"
    emotion_column: str = "emotion"