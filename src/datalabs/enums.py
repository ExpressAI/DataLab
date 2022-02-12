from dataclasses import dataclass, field
from typing import List
from enum import Enum



class PLMType(str, Enum):
    masked_language_model = "masked-language-model"
    left_to_right = "left-to-right"
    encoder_decoder = "encoder-decoder"

    @staticmethod
    def list():
        return list(map(lambda c: c.value, PLMType))



class SettingType(str, Enum):
    zero_shot = "zero-shot"
    few_shot = "few-shot"
    full_dataset = "full-dataset"