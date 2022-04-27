from typing import Optional, Iterable
from datalabs.info import SysOutputInfo, BucketPerformance, Performance, Table
from datalabs.utils import analysis
from datalabs.utils.analysis import *  # noqa
from datalabs.utils.eval_bucket import *  # noqa
from datalabs.metric import Accuracy  # noqa
from datalabs.metric import F1score  # noqa
from tqdm import tqdm
from datalabs.utils.feature_funcs import *  # noqa
from datalabs.utils.spacy_loader import spacy_loader
from typing import Iterator, Dict, List
from datalabs import load_dataset
from datalabs.operations.aggregate.text_classification import text_classification_aggregating

@text_classification_aggregating(name="get_statistics", contributor="datalab",
                                 task="text-classification",
                                 description="Calculate the overall statistics (e.g., average length) of "
                                             "a given text classification dataset")
def get_statistics(samples: Iterator):
    """
    Input:
    samples: [{
     "text":
     "label":
    }]
    """

    vocab = {}
    length_fre = {}
    for sample in tqdm(samples):
        text, label = sample["text"], sample["label"]
        length = len(text.split(" "))

        if length in length_fre.keys():
            length_fre[length] += 1
        else:
            length_fre[length] = 1

        # update vocabulary
        for w in text.split(" "):
            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 1

    # the rank of each word based on its frequency
    sorted_dict = {key: rank for rank, key in enumerate(sorted(set(vocab.values()), reverse=True), 1)}
    vocab_rank = {k: sorted_dict[v] for k, v in vocab.items()}


    for k, v in length_fre.items():
        length_fre[k] = length_fre[k]*1.0/len(samples)

    return {"vocab":vocab,
            "vocab_rank":vocab_rank,
            "length_fre":length_fre}


class TCExplainaboardBuilder:
    """
    Input: System Output file List[dict];  Metadata info
    Output: Analysis
    """

    def __init__(
        self,
        info: SysOutputInfo,
        system_output_object: Iterable[dict],
        feature_table: Optional[Table] = {},
        gen_kwargs: dict = None,
    ):
        self._info = info
        self._system_output: Iterable[dict] = system_output_object
        self.gen_kwargs = gen_kwargs
        self._data: Table = feature_table
        # _samples_over_bucket_true: Dict(feature_name, bucket_name, sample_id_true_label):
        # samples in different buckets
        self._samples_over_bucket = {}
        # _performances_over_bucket: performance in different bucket: Dict(feature_name, bucket_name, performance)
        self._performances_over_bucket = {}

        # Calculate statistics of training set
        self.statistics = None
        if None != self._info.dataset_name:
            try:
                dataset = load_dataset(self._info.dataset_name, self._info.sub_dataset_name)
                if len(dataset['train']._stat) == 0 or self._info.reload_stat == False: # calculate the statistics (_stat) when _stat is {} or `reload_stat` is False
                    new_train = dataset['train'].apply(get_statistics, mode = "local")
                    self.statistics = new_train._stat
                else:
                    self.statistics = dataset["train"]._stat
            except FileNotFoundError as err:
                eprint(
                    "The dataset hasn't been supported by DataLab so no training set dependent features will be supported by ExplainaBoard."
                    "You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md")


    @staticmethod
    def get_bucket_feature_value(feature_name: str):
        return "self._get_" + feature_name

    # define function for incomplete features
    def _get_sentence_length(self, existing_features: dict):
        return len(existing_features["text"].split(" "))

    def _get_token_number(self, existing_feature: dict):
        return len(existing_feature["text"])

    def _get_entity_number(self, existing_feature: dict):
        return len(
            spacy_loader.get_model("en_core_web_sm")(existing_feature["text"]).ents
        )

    def _get_label(self, existing_feature: dict):
        return existing_feature["true_label"]

    def _get_basic_words(self, existing_feature: dict):
        return get_basic_words(existing_feature["text"])  # noqa

    def _get_lexical_richness(self, existing_feature: dict):
        return get_lexical_richness(existing_feature["text"])  # noqa


    # training set dependent features
    def _get_num_oov(self, existing_features: dict):
        num_oov = 0

        for w in existing_features["text"].split(" "):
            if w not in self.statistics['vocab'].keys():
                num_oov += 1
        # print(num_oov)
        return num_oov


    # training set dependent features (this could be merged into the above one for further optimization)
    def _get_fre_rank(self, existing_features: dict):
        fre_rank = 0

        for w in existing_features["text"].split(" "):
            if w not in self.statistics['vocab_rank'].keys():
                fre_rank += len(self.statistics['vocab_rank'])
            else:
                fre_rank += self.statistics['vocab_rank'][w]

        fre_rank = fre_rank * 1.0 / len(existing_features["text"].split(" "))
        return fre_rank

    # training set dependent features
    def _get_length_fre(self, existing_features: dict):
        length_fre = 0
        length = len(existing_features["text"].split(" "))

        if length in self.statistics['length_fre'].keys():
            length_fre = self.statistics['length_fre'][length]


        return length_fre


    def _complete_feature(self):
        """
        This function is used to calculate features used for bucekting, such as sentence_length
        :param feature_table_iterator:
        :return:
        """
        # Get names of bucketing features
        # print(f"self._info.features.get_bucket_features()\n {self._info.features.get_bucket_features()}")
        bucket_features = self._info.features.get_bucket_features()
        for _id, dict_sysout in tqdm(
            enumerate(self._system_output), desc="featurizing"
        ):
            # Get values of bucketing features
            for bucket_feature in bucket_features:

                # this is need due to `del self._info.features[bucket_feature]`
                if bucket_feature not in self._info.features.keys():
                    continue
                # If there is a training set dependent feature while no pre-computed statistics for it,
                # then skip bucketing along this feature
                if self._info.features[bucket_feature].require_training_set and self.statistics == None:
                    del self._info.features[bucket_feature]
                    continue

                feature_value = eval(
                    TCExplainaboardBuilder.get_bucket_feature_value(bucket_feature)
                )(dict_sysout)
                dict_sysout[bucket_feature] = feature_value
            # if self._data is None:
            #     self._data = {}
            self._data[_id] = dict_sysout
            yield _id, dict_sysout

    def get_overall_performance(self):
        predicted_labels, true_labels = [], []

        for _id, feature_table in self._data.items():

            predicted_labels.append(feature_table["predicted_label"])
            true_labels.append(feature_table["true_label"])

        for metric_name in self._info.metric_names:
            one_metric = eval(metric_name)(
                true_labels=true_labels,
                predicted_labels=predicted_labels,
                is_print_confidence_interval=self._info.results.is_print_confidence_interval,
            )
            overall_value_json = one_metric.evaluate()

            overall_value = overall_value_json["value"]
            confidence_score_low = overall_value_json["confidence_score_low"]
            confidence_score_up = overall_value_json["confidence_score_up"]
            overall_performance = Performance(
                metric_name=metric_name,
                value=float(format(overall_value, '.4g')),
                confidence_score_low=float(format(confidence_score_low, '.4g')),
                confidence_score_up=float(format(confidence_score_up, '.4g')),
            )

            if self._info.results.overall is None:
                self._info.results.overall = {}
                self._info.results.overall[metric_name] = overall_performance
            else:
                self._info.results.overall[metric_name] = overall_performance

    def _bucketing_samples(self, sysout_iterator):

        sample_address = ""
        feature_to_sample_address_to_value = {}

        # Preparation for bucketing
        for _id, dict_sysout in sysout_iterator:

            sample_address = str(_id)  # this could be abstracted later
            for feature_name in self._info.features.get_bucket_features():
                # If there is a training set dependent feature while no pre-computed statistics for it,
                # then skip bucketing along this feature


                if feature_name not in feature_to_sample_address_to_value.keys():
                    feature_to_sample_address_to_value[feature_name] = {}
                else:
                    feature_to_sample_address_to_value[feature_name][
                        sample_address
                    ] = dict_sysout[feature_name]

        # Bucketing
        for feature_name in tqdm(
            self._info.features.get_bucket_features(), desc="bucketing"
        ):

            # print(f"Feature Name: {feature_name}\n"
            #       f"Bucket Hyper:\n function_name: {self._info.features[feature_name].bucket_info._method} \n"
            #       f"bucket_number: {self._info.features[feature_name].bucket_info._number}\n"
            #       f"bucket_setting: {self._info.features[feature_name].bucket_info._setting}\n")

            self._samples_over_bucket[feature_name] = eval(
                self._info.features[feature_name].bucket_info._method
            )(
                dict_obj=feature_to_sample_address_to_value[feature_name],
                bucket_number=self._info.features[feature_name].bucket_info._number,
                bucket_setting=self._info.features[feature_name].bucket_info._setting,
            )

            # print(f"self._samples_over_bucket.keys():\n{self._samples_over_bucket.keys()}")

            # evaluating bucket: get bucket performance
            self._performances_over_bucket[feature_name] = self.get_bucket_performance(
                feature_name
            )

    def get_bucket_performance(self, feature_name: str):
        """
        This function defines how to get bucket-level performance w.r.t a given feature (e.g., sentence length)
        :param feature_name: the name of a feature, e.g., sentence length
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        bucket_name_to_performance = {}
        for bucket_interval, sample_ids in self._samples_over_bucket[
            feature_name
        ].items():

            bucket_true_labels = []
            bucket_predicted_labels = []
            bucket_cases = []

            for sample_id in sample_ids:

                true_label = self._data[int(sample_id)]["true_label"]
                predicted_label = self._data[int(sample_id)]["predicted_label"]
                sent = self._data[int(sample_id)]["text"]  # noqa
                s_id = self._data[int(sample_id)]["id"]

                # get a bucket of true/predicted labels
                bucket_true_labels.append(true_label)
                bucket_predicted_labels.append(predicted_label)
                # get a bucket of cases (e.g., errors)
                if self._info.results.is_print_case:
                    if true_label != predicted_label:
                        # bucket_case = true_label + "|||" + predicted_label + "|||" + sent
                        # bucket_case = {"true_label":(s_id,["true_label"]),
                        #                "predicted_label":(s_id,["predicted_label"]),
                        #                "text":(s_id,["text"])}
                        bucket_case = str(s_id)
                        bucket_cases.append(bucket_case)

            bucket_name_to_performance[bucket_interval] = []
            for metric_name in self._info.metric_names:

                one_metric = eval(metric_name)(
                    true_labels=bucket_true_labels,
                    predicted_labels=bucket_predicted_labels,
                    is_print_confidence_interval=self._info.results.is_print_confidence_interval,
                )
                bucket_value_json = one_metric.evaluate()

                bucket_value = bucket_value_json["value"]
                confidence_score_low = bucket_value_json["confidence_score_low"]
                confidence_score_up = bucket_value_json["confidence_score_up"]



                bucket_performance = BucketPerformance(
                    bucket_name=bucket_interval,
                    metric_name=metric_name,
                    value=format(bucket_value, '.4g'),
                    confidence_score_low=format(confidence_score_low, '.4g'),
                    confidence_score_up=format(confidence_score_up, '.4g'),
                    n_samples=len(bucket_true_labels),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)  # noqa

    def _generate_report(self):
        dict_fine_grained = {}
        for feature_name, metadata in self._performances_over_bucket.items():
            dict_fine_grained[feature_name] = []
            for bucket_name, bucket_performance in metadata.items():
                bucket_name = analysis.beautify_interval(bucket_name)

                # instantiation
                dict_fine_grained[feature_name].append(bucket_performance)

        self._info.results.fine_grained = dict_fine_grained

    def _print_bucket_info(self):
        for feature_name in self._performances_over_bucket.keys():
            print_dict(  # noqa
                self._performances_over_bucket[feature_name], feature_name
            )

    def run(self) -> SysOutputInfo:
        eb_generator = self._complete_feature()
        self._bucketing_samples(eb_generator)
        self.get_overall_performance()
        self._print_bucket_info()
        self._generate_report()
        return self._info
