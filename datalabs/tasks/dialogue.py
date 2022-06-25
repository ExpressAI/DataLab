from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.dialogue)
@dataclass
class Dialogue(TaskTemplate):
    task: TaskType = TaskType.dialogue
    content_column: str = "content"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]
        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {"content": Sequence(Value("string"))}
            )


@register_task(TaskType.knowledge_driven_dialogue)
@dataclass
class KnowledgeDrivenDialogue(Dialogue):
    task: TaskType = TaskType.knowledge_driven_dialogue
    content_column: str = "content"
    knowledge_column: str = "knowledge"


# @register_task(TaskType.goal_oriented_knowledge_driven_dialogue)
# @dataclass
# class GoalOrientedKnowledgeDrivenDialogue(Dialogue):
#     task: TaskType = TaskType.goal_oriented_knowledge_driven_dialogue
#     goal_column: str = "goal"
#     content_column: str = "content"
#     knowledge_column: str = "knowledge"
#     response_column: str = "response"
#
#     input_schema: ClassVar[Features] = Features(
#         {
#             "goal": Sequence(Value("string")),
#             "content": Sequence(Value("string")),
#             "knowledge": Sequence(Value("string")),
#             "response": Value("string"),
#         }
#     )


@register_task(TaskType.task_oriented_dialogue)
@dataclass
class TaskOrientedDialogue(Dialogue):
    task: TaskType = TaskType.task_oriented_dialogue
    content_column: str = "content"


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
