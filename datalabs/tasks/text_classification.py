from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

from datalabs.enums import Metrics, PLMType, PromptShape, SignalType
from datalabs.features import ClassLabel, Features, Value
from datalabs.prompt import Prompt, PromptResult
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.text_classification)
@dataclass
class TextClassification(TaskTemplate):
    task: TaskType = TaskType.text_classification
    text_column: str = "text"
    label_column: str = "label"
    labels: Optional[Tuple[str]] = None

    def set_labels(self, labels):
        self.__dict__["labels"] = tuple(labels)
        self.__dict__["label_schema"] = self.label_schema.copy()
        self.label_schema["labels"] = ClassLabel(names=labels)

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]

        self.input_schema: ClassVar[Features] = Features(
            {
                self.text_column: Value("string"),
            }
        )
        self.label_schema: ClassVar[Features] = Features(
            {self.label_column: ClassLabel}
        )

        if self.labels:
            if len(self.labels) != len(set(self.labels)):
                raise ValueError("Labels must be unique")
            # Cast labels to tuple to allow hashing
            self.__dict__["labels"] = tuple(self.labels)
            self.__dict__["label_schema"] = self.label_schema.copy()
            self.label_schema["labels"] = ClassLabel(names=self.labels)


@register_task(TaskType.sentiment_classification)
@dataclass
class SentimentClassification(TextClassification):
    task: TaskType = TaskType.sentiment_classification
    text_column: str = "text"
    label_column: str = "label"


@register_task(TaskType.emotion_classification)
@dataclass
class EmotionClassification(TextClassification):
    task: TaskType = TaskType.emotion_classification
    text_column: str = "text"
    label_column: str = "label"


@register_task(TaskType.hatespeech_identification)
@dataclass
class HatespeechIdentification(TextClassification):
    task: TaskType = TaskType.hatespeech_identification
    text_column: str = "text"
    label_column: str = "label"


@register_task(TaskType.toxicity_identification)
@dataclass
class ToxicityIdentification(TextClassification):
    task: TaskType = TaskType.toxicity_identification
    text_column: str = "text"
    label_column: str = "label"


@register_task(TaskType.next_token_classification)
@dataclass
class NextTokenClassification(TextClassification):
    task: TaskType = TaskType.next_token_classification
    text_column: str = "text"
    label_column: str = "label"


@register_task(TaskType.question_classification)
@dataclass
class QuestionClassification(TextClassification):
    task: TaskType = TaskType.question_classification
    text_column: str = "text"
    label_column: str = "label"


@register_task(TaskType.spam_identification)
@dataclass
class SpamClassification(TextClassification):
    task: TaskType = TaskType.spam_identification
    text_column: str = "text"
    label_column: str = "label"


@register_task(TaskType.intent_classification)
@dataclass
class IntentClassification(TextClassification):
    task: TaskType = TaskType.intent_classification
    text_column: str = "text"
    label_column: str = "label"


@register_task(TaskType.grammatical_judgment)
@dataclass
class GrammaticalJudgment(TextClassification):
    task: TaskType = TaskType.grammatical_judgment
    text_column: str = "text"
    label_column: str = "label"


@register_task(TaskType.topic_classification)
@dataclass
class TopicClassification(TextClassification):
    task: TaskType = TaskType.topic_classification
    text_column: str = "text"
    label_column: str = "label"

    results = [
        PromptResult(value=0.0, plm="bert-base-uncased", metric=Metrics.accuracy.value),
        PromptResult(
            value=0.0, plm="facebook/bart-large", metric=Metrics.accuracy.value
        ),
        PromptResult(value=0.0, plm="t5-11b", metric=Metrics.accuracy.value),
    ]

    prompts_raw = [
        Prompt(
            template="Given the text: {{text}}, is it about"
            " {{textual_choices_with_or}}? ||| {{answers[label]}}",
            description="We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[
                PLMType.encoder_decoder.value,
                PLMType.left_to_right.value,
            ],
            signal_type=[SignalType.topic_classification.value],
            features={
                "shape": PromptShape.prefix.value,
                "length": len(
                    (
                        "Given the text: {{text}}, is it about {"
                        "{textual_choices_with_or}}? ||| {{answers[label]}}"
                    ).split(" ")
                ),
                "skeleton": "task-level prompts",
            },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/",
        ),
        Prompt(
            template="Given the text: {{text}}, it is about "
            "[mask]. ||| {{answers[label]}}",
            description="We use [mask] to represent the mask symbol"
            " from a given PLM's vocabulary. "
            "We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[PLMType.masked_language_model.value],
            signal_type=[SignalType.topic_classification.value],
            features={
                "shape": PromptShape.cloze.value,
                "length": len(
                    "Given the text: {{text}}, it is about "
                    "[mask]. ||| {{answers[label]}}".split(" ")
                ),
                "skeleton": "task-level prompts",
            },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/",
        ),
        Prompt(
            template="Given the text: {{text}} Classify this text."
            " You may choose from"
            " {{textual_choices_without_or}}. ||| {{answers[label]}}",
            description="We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[
                PLMType.encoder_decoder.value,
                PLMType.left_to_right.value,
            ],
            signal_type=[SignalType.topic_classification.value],
            features={
                "shape": PromptShape.prefix.value,
                "length": len(
                    (
                        "Given the text: {{text}} Classify "
                        "this text. You may choose from"
                        " {{textual_choices_without_or}}. ||| {{answers[label]}}"
                    ).split(" ")
                ),
                "skeleton": "task-level prompts",
            },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/",
        ),
        Prompt(
            template="Given the text: {{text}} Given a list"
            " of categories: {{textual_choices_without_or}},"
            " what category does the paragraph "
            "belong to? ||| {{answers[label]}}",
            description="We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[
                PLMType.encoder_decoder.value,
                PLMType.left_to_right.value,
            ],
            signal_type=[SignalType.topic_classification.value],
            features={
                "shape": PromptShape.prefix.value,
                "length": len(
                    (
                        "Given the text: {{text}} Given "
                        "a list of categories: {{textual_choices_without_or}}, "
                        "what category does the paragraph belong "
                        "to? ||| {{answers[label]}}"
                    ).split(" ")
                ),
                "skeleton": "task-level prompts",
            },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/",
        ),
        Prompt(
            template="Given the text: {{text}} Pick one category "
            "for the previous text. The options are"
            " {{textual_choices_without_or}}. ||| {{answers[label]}}",
            description="We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[
                PLMType.encoder_decoder.value,
                PLMType.left_to_right.value,
            ],
            signal_type=[SignalType.topic_classification.value],
            features={
                "shape": PromptShape.prefix.value,
                "length": len(
                    (
                        "Given the text: {{text}} Pick one category"
                        " for the previous text. The options are"
                        " {{textual_choices_without_or}}. ||| {{answers[label]}}"
                    ).split(" ")
                ),
                "skeleton": "task-level prompts",
            },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/",
        ),
        Prompt(
            template="Given the text: {{text}} Can you identify"
            "the category of this text? "
            "{{textual_choices_with_or}}? ||| {{answers[label]}}",
            description="We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[
                PLMType.encoder_decoder.value,
                PLMType.left_to_right.value,
            ],
            signal_type=[SignalType.topic_classification.value],
            features={
                "shape": PromptShape.prefix.value,
                "length": len(
                    (
                        "Given the text: {{text}} Can you identify "
                        "the category of this text?"
                        " {{textual_choices_with_or}}? ||| {{answers[label]}}"
                    ).split(" ")
                ),
                "skeleton": "task-level prompts",
            },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/",
        ),
        Prompt(
            template="Given the text: {{text}} What\\'s the main"
            " topic of this paragraph? "
            "{{textual_choices_with_or}}? ||| {{answers[label]}}",
            description="We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[
                PLMType.encoder_decoder.value,
                PLMType.left_to_right.value,
            ],
            signal_type=[SignalType.topic_classification.value],
            features={
                "shape": PromptShape.prefix.value,
                "length": len(
                    (
                        "Given the text: {{text}} What\\'s the main"
                        " topic of this paragraph?"
                        " {{textual_choices_with_or}}? ||| {{answers[label]}}"
                    ).split(" ")
                ),
                "skeleton": "task-level prompts",
            },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/",
        ),
        Prompt(
            template="Given the text: {{text}} Is this "
            "a piece of text regarding "
            "{{textual_choices_with_or}}? ||| {{answers[label]}}",
            description="We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[
                PLMType.encoder_decoder.value,
                PLMType.left_to_right.value,
            ],
            signal_type=[SignalType.topic_classification.value],
            features={
                "shape": PromptShape.prefix.value,
                "length": len(
                    (
                        "Given the text: {{text}} Is this"
                        " a piece of text regarding "
                        "{{textual_choices_with_or}}? ||| {{answers[label]}}"
                    ).split(" ")
                ),
                "skeleton": "task-level prompts",
            },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/",
        ),
    ]
    prompts = {x.id: x for x in prompts_raw}


@register_task(TaskType.question_answering_classification)
@dataclass
class QuestionAnsweringClassification(TextClassification):
    task: TaskType = TaskType.question_answering_classification
    text_column: str = "text"
    label_column: str = "label"

    input_schema: ClassVar[Features] = Features(
        {
            "text": {
                "question": Value("string"),
                "description": Value("string"),
                "answers": Value("string"),
            }
        }
    )
