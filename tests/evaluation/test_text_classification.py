import os
import pathlib
import unittest

from datalabs import FileType, get_loader, get_processor, load_dataset, Source, TaskType
from datalabs.operations.aggregate.auto_eval import explainaboard
from datalabs.operations.infer.inference import inference

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class MyTestCase(unittest.TestCase):
    def test_evaluation(self):

        path_data = artifacts_path + "test-classification.tsv"
        loader = get_loader(
            TaskType.text_classification,
            Source.local_filesystem,
            FileType.tsv,
            path_data,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.text_classification.value,
            "dataset_name": "sst2",
            "metric_names": ["Accuracy"],
        }

        processor = get_processor(TaskType.text_classification, metadata, data)
        # self.assertEqual(len(processor._features), 4)

        analysis = processor.process()
        # analysis.write_to_directory("./")


if __name__ == "__main__":
    unittest.main()
