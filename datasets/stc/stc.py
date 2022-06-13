import json
import os

import datalabs
from datalabs import get_task, TaskType


_CITATION = """\
@article{shang2015neural,
  title={Neural responding machine for short-text conversation},
  author={Shang, Lifeng and Lu, Zhengdong and Li, Hang},
  journal={arXiv preprint arXiv:1503.02364},
  year={2015}
}
"""

_DESCRIPTION = """\

"""
_TRAIN_VALIDATION_DOWNLOAD_URL="https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/STC/STC.json"
_TEST_DOWNLOAD_URL="https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/STC/STC_test.json"


class STC(datalabs.GeneratorBasedBuilder):
   
    VERSION = datalabs.Version("1.0.0")
    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "source": datalabs.Value("string"),
                    "reference": datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
            languages = ["zh"],
            task_templates=[
                get_task(TaskType.single_turn_dialogue)(
                    source_column="source", reference_column="reference"
                )
            ],
        )

        

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_validation_path = dl_manager.download_and_extract(_TRAIN_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)


        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_validation_path,'split':'train'}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": train_validation_path,'split':"valid"}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path,'split':'test'})
        ]

    def _generate_examples(self, filepath,split):
        """This function returns the examples."""

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for id_, line in enumerate(data[split]):
                yield id_, {"source": line[0], "reference": line[1]}