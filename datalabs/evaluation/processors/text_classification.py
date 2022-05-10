from typing import Iterable
from datalabs import features
from datalabs.tasks.task_info import TaskType
from .processor import Processor
from .processor_registry  import register_processor
from ..builders.text_classification import TCExplainaboardBuilder




@register_processor(TaskType.text_classification)
class TextClassificationProcessor(Processor):
    _task_type = TaskType.text_classification
    _features = features.Features(
        {
            "text": features.Value("string"),
            "true_label": features.ClassLabel(names=["1", "0"], is_bucket=False),
            "predicted_label": features.ClassLabel(names=["1", "0"], is_bucket=False),
            "label": features.Value(
                dtype="string",
                description="category",
                is_bucket=True,
                bucket_info=features.BucketInfo(
                    _method="bucket_attribute_discrete_value", _number=4, _setting=1
                ),
            ),
            "sentence_length": features.Value(
                dtype="float",
                description="text length",
                is_bucket=True,
                bucket_info=features.BucketInfo(
                    _method="bucket_attribute_specified_bucket_value",
                    _number=4,
                    _setting=(),
                ),
            ),
            "token_number": features.Value(
                dtype="float",
                description="the number of chars",
                is_bucket=True,
                bucket_info=features.BucketInfo(
                    _method="bucket_attribute_specified_bucket_value",
                    _number=4,
                    _setting=(),
                ),
            ),
            "basic_words": features.Value(
                dtype="float",
                description="the ratio of basic words",
                is_bucket=True,
                bucket_info=features.BucketInfo(
                    _method="bucket_attribute_specified_bucket_value",
                    _number=4,
                    _setting=(),
                ),
            ),
            "lexical_richness": features.Value(
                dtype="float",
                description="lexical diversity",
                is_bucket=True,
                bucket_info=features.BucketInfo(
                    _method="bucket_attribute_specified_bucket_value",
                    _number=4,
                    _setting=(),
                ),
            ),
            "entity_number": features.Value(
                dtype="float",
                description="the number of entities",
                is_bucket=True,
                bucket_info=features.BucketInfo(
                    _method="bucket_attribute_specified_bucket_value",
                    _number=4,
                    _setting=(),
                ),
            ),
            "num_oov": features.Value(
                dtype="float",
                description="the number of out-of-vocabulary words",
                is_bucket=True,
                bucket_info=features.BucketInfo(
                    _method="bucket_attribute_specified_bucket_value",
                    _number=4,
                    _setting=(),
                ),
                require_training_set=True,
            ),
            "fre_rank": features.Value(
                dtype="float",
                description="the average rank of each work based on its frequency in training set",
                is_bucket=True,
                bucket_info=features.BucketInfo(
                    _method="bucket_attribute_specified_bucket_value",
                    _number=4,
                    _setting=(),
                ),
                require_training_set=True,
            ),
            "length_fre": features.Value(
                dtype="float",
                description="the frequency of text length in training set",
                is_bucket=True,
                bucket_info=features.BucketInfo(
                    _method="bucket_attribute_specified_bucket_value",
                    _number=4,
                    _setting=(),
                ),
                require_training_set=True,
            ),

        }
    )

    def __init__(self, metadata: dict, system_output_data: Iterable[dict]) -> None:
        if metadata is None:
            metadata = {}
        if "task_name" not in metadata.keys():
            metadata["task_name"] = TaskType.text_classification.value
        if "metric_names" not in metadata.keys():
            metadata["metric_names"] = ["Accuracy"]
        super().__init__(metadata, system_output_data)
        self._builder = TCExplainaboardBuilder(
            self._system_output_info, system_output_data
        )
