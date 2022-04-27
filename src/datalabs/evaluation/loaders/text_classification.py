from typing import Dict, Iterable, List
from datalabs.constants import Source, FileType
from enum import Enum
from datalabs.tasks.task_info import TaskType
from .loader import register_loader
from .loader import Loader


@register_loader(TaskType.text_classification)
class TextClassificationLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    text \t true_label \t predicted_label

    usage:
        please refer to `test_loaders.py`
    """

    def __init__(self, source: Source, file_type: Enum, data: str = None):

        if source is None:
            source = Source.local_filesystem
        if file_type is None:
            file_type = FileType.tsv

        self._source = source
        self._file_type = file_type
        self._data = data

    def load(self) -> Iterable[Dict]:
        """
        :param path_system_output: the path of system output file with following format:
        text \t label \t predicted_label
        :return: class object
        """
        raw_data = self._load_raw_data_points()
        data: List[Dict] = []
        if self._file_type == FileType.tsv:
            for id, dp in enumerate(raw_data):
                text, true_label, predicted_label = dp[:3]
                data.append(
                    {
                        "id": str(id),
                        "text": text.strip(),
                        "true_label": true_label.strip(),
                        "predicted_label": predicted_label.strip(),
                    }
                )
        elif self._file_type == FileType.json:
            for id, info in enumerate(raw_data):
                text, true_label, predicted_label = (
                    info["text"],
                    info["true_label"],
                    info["predicted_label"],
                )
                data.append(
                    {
                        "id": str(id),
                        "text": text.strip(),
                        "true_label": true_label.strip(),
                        "predicted_label": predicted_label.strip(),
                    }
                )
        elif self._file_type == FileType.datalab:
            for id, info in enumerate(raw_data):
                text, true_label, predicted_label = (
                    info["text"],
                    info["label"],
                    info["prediction"],
                )

                data.append(
                    {
                        "id": str(id),
                        "text": text.strip(),
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                    }
                )
        else:
            raise NotImplementedError
        return data
