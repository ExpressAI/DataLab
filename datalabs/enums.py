from enum import Enum


class PLMType(str, Enum):
    masked_language_model: str = "masked-language-model"
    left_to_right: str = "left-to-right"
    encoder_decoder: str = "encoder-decoder"

    @staticmethod
    def list():
        return list(map(lambda c: c.value, PLMType))


class SettingType(str, Enum):
    zero_shot: str = "zero-shot"
    few_shot: str = "few-shot"
    full_dataset: str = "full-dataset"


class SignalType(str, Enum):
    text_compression: str = "text-compression"
    text_summarization: str = "text-summarization"
    text_classification: str = "text-classification"
    topic_classification: str = "topic-classification"


class PromptShape(str, Enum):
    prefix: str = "prefix"
    cloze: str = "cloze"


class Metrics(str, Enum):
    accuracy: str = "accuracy"
    rouge1: str = "rouge1"
    rouge2: str = "rouge2"
    rougeL: str = "rougeL"
    f1: str = "f1"
    bleu: str = "bleu"
