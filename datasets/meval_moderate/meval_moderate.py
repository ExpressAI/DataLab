
import json
import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
Dataset for Evaluating Moderate Classifier.
"""

_CITATION = """\
No
"""

_HOMEPAGE = ""


fields = ["toxicity", "identity_attack", "insult", "obscene", "sexual_explicit", "threat"]

_URLs = "https://storage.googleapis.com/inspired-public-data/datasets/meval_moderate/processed/{}/meta_test.json"
# _URLs = "https://raw.githubusercontent.com/ShiinaHiiragi/multi-task-dataset/master/{}.task.{}"

class MEvalModerate(datalabs.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(
            name="{}".format(domain), version=datalabs.Version("1.0.0")
        )
        for domain in fields
    ]


    def _info(self):
        features = datalabs.Features(
            {
                "text": datalabs.Value("string"),
                "label": datalabs.features.ClassLabel(names=["normal", "abnormal"]),
            }
        )
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datalab page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license="CC-SA-3.0",
            # Citation for the dataset
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.text_classification)(
                    text_column="text", label_column="label"
                )
            ],
            languages=["en"],
        )

    def _split_generators(self, dl_manager):
        domain = str(self.config.name)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": dl_manager.download_and_extract(_URLs.format(domain)),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        # read from jsonl
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                text = data["text"]
                label = data["label"]
                yield id_, {"text": text, "label": label}




