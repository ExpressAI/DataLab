import sys


feature_map_basic = {
    "avg_length": "the average length",
    "avg_basic_words": "the ratio of basic words (pre-defied by a dictionary)",
    "avg_lexical_richness": "the lexical diversity",
    "avg_gender_bias_word_male": "the average ratio of male words",
    "avg_gender_bias_word_female": "the average ratio of female words",
    "avg_gender_bias_single_name_male": "the average ratio of male names",
    "avg_gender_bias_single_name_female": "the average ratio of female names",
    "avg_gender_bias_name_male": "the average ratio of male names",
    "avg_gender_bias_name_female": "the average ratio of female names",
    "avg_span_length_of_ner_tags": "the average of entity length",
    "avg_eCon_of_ner_tags": "the average of entity's label consistency "
    "(defined in the paper: Interpretable Multi-dataset"
    " Evaluation for Named Entity Recognition)",
    "avg_eFre_of_ner_tags": "the average of entity frequency",
    "avg_density": "the average density (measures to what extent a summary "
    "covers the content in the source text)",
    "avg_coverage": "the average of coverage (measures to what extent a summary"
    " covers the content in the source text)",
    "avg_compression": "the average of compression (measures the compression"
    " ratio from the source text to the generated summary)",
    "avg_repetition": "the average of repetition (measures the rate of repeated "
    "segments in summaries. The segments are instantiated "
    "as trigrams)",
    "avg_novelty": "the average of novelty (the proportion of segments in the "
    "summaries that have not appeared in source documents. "
    "The segments are instantiated as bigrams)",
    "avg_copy_length": "the average of copy_length (the average length of segments "
    "in summary copied from source document)",
}

feature_map_basic2 = {
    "divide": "fraction",
    "minus": "difference",
    "add": "addition",
}


def get_feature_description(feature_name: str):

    desc = ""

    if ("avg_" + feature_name.split("avg_")[-1]) in feature_map_basic.keys():
        raw_feature_name = feature_name.split("_")[0]
        split_name = feature_name.split("_")[1]
        desc = feature_map_basic["avg_" + feature_name.split("avg_")[-1]]
        desc = desc + " of " + raw_feature_name + " in " + split_name + " set"

    elif feature_name.split("_")[0] == "bleu":
        raw_feature_name1 = feature_name.split("_")[1]
        raw_feature_name2 = feature_name.split("_")[2]
        split_name = feature_name.split("_")[-1]

        desc = (
            "the similarity score (using BLEU) between `"
            + raw_feature_name1
            + "` and `"
            + raw_feature_name2
            + "` in "
            + split_name
            + " set"
        )

    elif feature_name.split("_")[2] in feature_map_basic2.keys():
        operation_name = feature_map_basic2[feature_name.split("_")[2]]
        raw_feature_name1 = feature_name.split("_")[0]
        raw_feature_name2 = feature_name.split("_")[3]
        split_name = feature_name.split("_")[-2]

        desc = (
            "the length "
            + operation_name
            + " between `"
            + raw_feature_name1
            + "` and `"
            + raw_feature_name2
            + "` in "
            + split_name
            + " set"
        )

    elif "_".join(feature_name.split("_")[0:-1]) in feature_map_basic.keys():
        split_name = feature_name.split("_")[-1]
        desc = feature_map_basic["_".join(feature_name.split("_")[0:-1])]
        desc = desc + " in " + split_name + " set"
    elif feature_name.split("_of")[0] in feature_map_basic.keys():
        split_name = feature_name.split("_of")[-1].split("_")[1]

        desc = (
            feature_map_basic[feature_name.split("_of")[0]]
            + " of "
            + split_name
            + " set"
        )

    if desc == "":
        desc = feature_name

    return desc


# Usage & Test Cases:
# feature_name = "background_train_avg_gender_bias_single_name_male"
# feature_name = "bleu_question_situation_avg_test"
# feature_name = "question_length_divide_situation_avg_validation_length"
# feature_name = "avg_span_length_of_ner_tags_test"
# feature_name = "avg_eCon_of_ner_tags_validation"
# feature_name = "avg_compression_of_test_highlights_and_article"
# feature_name = "avg_copy_length_of_test_highlights_and_article"
# feature_name = "premise_length_add_hypothesis_avg_validation_length"
# feature_name = "premise_length_divide_hypothesis_avg_train_length"
# feature_name = "bleu_question_context_avg_train"
# feature_name = "question_length_divide_context_avg_validation_length"
#
# print(get_feature_description(feature_name))
