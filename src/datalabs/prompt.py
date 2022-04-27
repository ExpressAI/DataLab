from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional, Tuple, List
import json
import hashlib  # for mdb ids of prompts
import requests


@dataclass
class PromptResult:
    setting = "zero-shot"
    value: float = 0.0
    plm: str = None
    metric: str = None


"""Example
{
      "language": "en",
      "template": "{Text}, Overall it is a {Answer} movie.",
      "answer": {
        "positive": ["fantastic", "interesting"],
        "negative": ["boring"]
      },
      "supported_plm_types": ["masked_lm", "left_to_right", "encoder_decoder"],
      "results": [
        {
          "plm": "BERT",
          "metric": "accuracy",
          "setting": "zero-shot",
          "value": "87"
        },
        {
          "plm": "BART",
          "metric": "accuracy",
          "setting": "zero-shot",
          "value": "80"
        }

      ]
    }
"""


@dataclass
class Prompt:
    id: str = "null"  # this will be automatically assigned
    language: str = "en"
    description: str = "prompt description"
    template: str = None
    # in Prompt class, we define `answer` field as the mapping from the category name to a list of answer words.
    # for example answers={'World': ['World News','World Report']}, {'Sports': ['Sports']}, {'Business': ['Business']}, {'Science and Technology': ['Science and Technology']}
    answers: dict = None
    supported_plm_types: List[str] = None
    signal_type: List[str] = None
    # results: List[PromptResult] = None
    results: List[PromptResult] = None
    # features:Optional[Features] = None # {"length":Value("int64"), "shape":Value("string"), "skeleton": Value("string")}
    features: Optional[dict] = None  # {"length":5, "shape":"prefix", "skeleton": "what_about"}
    reference: str = None
    contributor: str = "Datalab"

    def __post_init__(self):
        # Convert back to the correct classes when we reload from dict
        if self.template is not None and self.answers is not None:
            if isinstance(self.answers, dict):
                self.id = hashlib.md5((self.template + json.dumps(self.answers)).encode()).hexdigest()
            if isinstance(self.answers, str):
                self.id = hashlib.md5((self.template + self.answers).encode()).hexdigest()
            else:
                self.id = hashlib.md5(self.template.encode()).hexdigest()
        else:
            self.id = hashlib.md5(self.template.encode()).hexdigest()


class Prompts:
    @classmethod
    def from_url(cls, URL):
        res = requests.get(URL)
        prompts = json.loads(res.text)
        # new_prompts = {x["id"]: Prompt(**x) for x in prompts}
        # prompts = []
        # for dic in dics:
        #     prompts.append(Prompt(**dic))
        return prompts
