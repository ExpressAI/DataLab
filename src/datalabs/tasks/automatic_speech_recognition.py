# coding=utf-8
# Copyright 2022 The HuggingFace Datasets, DataLab Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import ClassVar, Dict

from ..features import Features, Value
from .base import TaskTemplate


@dataclass
class AutomaticSpeechRecognition(TaskTemplate):
    task_category:str = "automatic-speech-recognition"
    task: str = "automatic-speech-recognition"
    # TODO(lewtun): Replace input path feature with dedicated `Audio` features
    # when https://github.com/huggingface/datasets/pull/2324 is implemented
    input_schema: ClassVar[Features] = Features({"audio_file_path": Value("string")})
    label_schema: ClassVar[Features] = Features({"transcription": Value("string")})
    audio_file_path_column: str = "audio_file_path"
    transcription_column: str = "transcription"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.audio_file_path_column: "audio_file_path", self.transcription_column: "transcription"}
