"""TODO(race): Add a description here."""


import json

import datalabs
from datalabs.tasks import QuestionAnsweringChoiceWithContext

_CITATION = """\
@article{lai2017large,
    title={RACE: Large-scale ReAding Comprehension Dataset From Examinations},
    author={Lai, Guokun and Xie, Qizhe and Liu, Hanxiao and Yang, Yiming and Hovy, Eduard},
    journal={arXiv preprint arXiv:1704.04683},
    year={2017}
}
"""

_DESCRIPTION = """\
Race is a large-scale reading comprehension dataset with more than 28,000 passages and nearly 100,000 questions. The
 dataset is collected from English examinations in China, which are designed for middle school and high school students.
The dataset can be served as the training and test sets for machine comprehension.
"""

_URL = "http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz"


class Race(datalabs.GeneratorBasedBuilder):
    """ReAding Comprehension Dataset From Examination dataset from CMU"""

    VERSION = datalabs.Version("0.1.0")

    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(name="high", description="Exams designed for high school students", version=VERSION),
        datalabs.BuilderConfig(
            name="middle", description="Exams designed for middle school students", version=VERSION
        ),
        datalabs.BuilderConfig(
            name="all", description="Exams designed for both high school and middle school students", version=VERSION
        ),
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datalabs.Features(
                {
                    "example_id": datalabs.Value("string"),
                    "context": datalabs.Value("string"), # context ->article
                    "question": datalabs.Value("string"),
                    # "answers": datalabs.Value("string"),  # answers ->answer
                    "answers": # answers -> answer
                        {
                            "text": datalabs.features.Sequence(datalabs.Value("string")),
                            "answer_start": datalabs.features.Sequence(datalabs.Value("int32")),
                        },
                    "options": datalabs.features.Sequence(datalabs.Value("string"))
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="http://www.cs.cmu.edu/~glai1/data/race/",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringChoiceWithContext(
                    question_column="question", context_column="context", answers_column="answers", options_column="options"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        archive = dl_manager.download(_URL)
        case = str(self.config.name)
        if case == "all":
            case = ""
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"train_test_or_eval": f"RACE/test/{case}", "files": dl_manager.iter_archive(archive)},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"train_test_or_eval": f"RACE/train/{case}", "files": dl_manager.iter_archive(archive)},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"train_test_or_eval": f"RACE/dev/{case}", "files": dl_manager.iter_archive(archive)},
            ),
        ]

    def _generate_examples(self, train_test_or_eval, files):
        """Yields examples."""
        for file_idx, (path, f) in enumerate(files):
            if path.startswith(train_test_or_eval) and path.endswith(".txt"):
                data = json.loads(f.read().decode("utf-8"))
                questions = data["questions"]
                answers = data["answers"]
                options = data["options"]
                for i in range(len(questions)):
                    question = questions[i]
                    answer = answers[i]
                    option = options[i]
                    # answers = [example["answer"].strip()]
                    yield f"{file_idx}_{i}", {
                        "example_id": data["id"],
                        "context": data["article"],
                        "question": question,
                        "answers": {
                            "answer_start": [-1] * len(answers),
                            "text": [answer],
                        },
                        # "answers": answer,
                        "options": option,
                    }