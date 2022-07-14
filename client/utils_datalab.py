from dataclasses import asdict
import multiprocessing

# this is task-dependent
from datalabs import load_dataset
from datalabs.operations.aggregate.text_classification import (
    get_features_dataset_level as get_features_dataset_level_text_classification,
)
from datalabs.utils.more_features import get_features_dataset, prefix_dict_key


def get_paper_template(
    year=None,
    venue=None,
    title=None,
    author=None,
    url=None,
    bib=None,
):
    return {
        "year": year,
        "venue": venue,
        "title": title,
        "author": author,
        "url": url,
        "bib": bib,
    }


def get_annotator_demographic_template(
    gender=None,
    race=None,
    native_language=None,
    socioeconomic_status=None,
    number_of_different_speakers_represented=None,
    presence_of_disordered_speech=None,
    training_in_linguistics=None,
):
    return {
        "gender": gender,
        "race_ethnicity": race,
        "native_language": native_language,
        "socioeconomic_status": socioeconomic_status,
        "number_of_different_speakers_represented": number_of_different_speakers_represented,  # noqa
        "presence_of_disordered_speech": presence_of_disordered_speech,
        "training_in_linguistics": training_in_linguistics,
    }


def get_speaker_demographic_template(
    gender=None,
    race=None,
    native_language=None,
    socioeconomic_status=None,
    number_of_different_speakers_represented=None,
    presence_of_disordered_speech=None,
):
    return {
        "gender": gender,
        "race_ethnicity": race,
        "native_language": native_language,
        "socioeconomic_status": socioeconomic_status,
        "number_of_different_speakers_represented": number_of_different_speakers_represented,  # noqa
        "presence_of_disordered_speech": presence_of_disordered_speech,
    }


def get_speech_situation_templates(
    time=None, place=None, modality=None, intended_audience=None
):
    return {
        "time": time,
        "place": place,
        "modality": modality,
        "intended_audience": intended_audience,
    }


def get_popularity_template(
    number_of_download=None,
    number_of_likes=None,
    number_of_reposts=None,
    number_of_visits=None,
):
    return {
        "number_of_download": number_of_download,
        "number_of_likes": number_of_likes,
        "number_of_reposts": number_of_reposts,
        "number_of_visits": number_of_visits,
    }


def get_size_template(samples=None, storage=None):
    return {
        "sample": samples,
        "storage": storage,
    }


def get_transformation_template(type):
    return {"type": type}


def get_info(
    dataset_name: str,
    sub_dataset_name_sdk: str,
    calculate_features=False,
    processed_list=None,
):
    """
    Input:
    dataset_name: the dataset name of dataloader script, for example, mr
    field: the field to be featurized
    Output:
    asdict(dataset['train']._info): metadata information
    features_mongodb: features of metadata information
    dataset: detailed sample of all dataset splits
    """

    # load dataset
    dataset = (
        load_dataset(dataset_name)
        if sub_dataset_name_sdk is None
        else load_dataset(dataset_name, sub_dataset_name_sdk)
    )

    # get split
    all_splits = list(dataset["train"]._info.splits.keys())

    # task
    # task_category  = dataset['train']._info.task_templates[0].task_category

    # get raw features
    raw_features = asdict(dataset[all_splits[0]]._info)["features"]

    features_mongodb = {}

    # from featurize.text_matching import get_features_sample_level

    for split_name in all_splits:
        # processed_list = [
        #     (tokenize, "text"),
        #     (feature_func, None)
        # ]

        # get sample-level advanced features
        prefix_name = ""
        for processed_func, processed_field in processed_list:
            if processed_field is not None:
                processed_func.processed_fields[0] = processed_field
                prefix_name = processed_field
            else:
                prefix_name = ""

            dataset[split_name] = dataset[split_name].apply(
                processed_func,
                num_proc=multiprocessing.cpu_count(),
                mode="memory",
                prefix=prefix_name,
            )
            print(dataset[split_name])

        # dataset[split_name] = dataset[split_name].apply(
        #     feature_func, num_proc=multiprocessing.cpu_count(), mode="memory"
        # )
        all_features = asdict(dataset[split_name]._info)["features"]

        # turn on advanced fields
        for feature_name, feature_info in all_features.items():
            # this is defined for the case when feature is `text_tokenized`
            if (
                feature_name.find("tokenize") != -1
                and all_features[feature_name]["dtype"] == "string"
            ):
                feature_info["raw_feature"] = True
                feature_info["is_bucket"] = False
            elif feature_name not in raw_features.keys():
                feature_info["raw_feature"] = False
                feature_info["is_bucket"] = True

        # add sample-level features
        features_mongodb.update(all_features)

    for split_name in all_splits:

        # if split_name == "train":
        #     continue

        # calculate dataset-level features
        dataset[split_name] = dataset[split_name].apply(
            get_features_dataset_level_text_classification, mode="memory", prefix="avg"
        )

        features_dataset = get_features_dataset(dataset[split_name]._stat)

        for attr, feat_info in features_dataset.items():

            feat_info = asdict(feat_info)
            value = dataset[split_name]._stat[attr]
            feat_info["value"] = value
            features_dataset[attr] = feat_info
        features_dataset_new = prefix_dict_key(features_dataset, prefix=split_name)

        # add dataset-level features
        features_mongodb.update(features_dataset_new)

    return asdict(dataset["train"]._info), features_mongodb, dataset


def validate_generate_db_metadata(
    dataset_name,
    transformation,
    version,
    task_categories,
    tasks,
    split,
    languages,
    sub_dataset=None,
    summary=None,
    homepage=None,
    repository=None,
    leaderboard=None,
    person_of_contact=None,
    features=None,
    speaker_demographic=None,
    annotator_demographic=None,
    speech_situation=None,
    size=None,
    license_=None,
    huggingface_link=None,
    curation_rationale=None,
    genre=None,
    quality=None,
    similar_datasets=None,
    popularity=None,
    creator_name=None,
    multilinguality=None,
    paper_info=None,
    prompt_infos=None,
    submitter_name=None,
    system_metadata_ids=None,
    data_typology=None,
    statistical_information=None,
):
    if dataset_name is None or dataset_name == "":
        raise Exception("dataset_name should not be None or ''")
    if transformation is None:
        raise Exception("transformation should not be None")
    if version is None or version == "":
        raise Exception("version should not be None or ''")

    if tasks is None or (not isinstance(tasks, list)) or len(tasks) == 0:
        raise Exception(
            "tasks should not be None and this object should be list"
            " type and its length should not be zero"
        )
    if (
        task_categories is None
        or (not isinstance(task_categories, list))
        or len(task_categories) == 0
    ):
        raise Exception(
            "task_categories should not be None and this object should be"
            " list type and its length should not be zero"
        )

    if split is None or (not isinstance(split, dict)):
        raise Exception("split should not be None and this object should be dict type")

    if languages is None or (not isinstance(languages, list)) or len(languages) == 0:
        raise Exception(
            "languages should not be None and this object should be list"
            " type and its length should not be zero"
        )

    return {
        "dataset_name": dataset_name,
        "sub_dataset": sub_dataset,
        "split": split,
        "summary": summary,
        "homepage": homepage,
        "repository": repository,
        "paper_info": paper_info,
        "leaderboard": leaderboard,
        "person_of_contact": person_of_contact,
        "tasks": tasks,
        "task_categories": task_categories,
        "language": languages,
        "features": features,
        "speaker_demographic": speaker_demographic,
        "annotator_demographic": annotator_demographic,
        "speech_situation": speech_situation,
        "size": size,
        "license": license_,
        "huggingface_link": huggingface_link,
        "curation_rationale": curation_rationale,
        "genre": genre,
        "quality": quality,
        "similar_datasets": similar_datasets,
        "creator_name": creator_name,
        "submitter_name": submitter_name,
        "multilinguality": multilinguality,
        "system_metadata_ids": system_metadata_ids,
        "popularity": popularity,
        "transformation": transformation,
        "version": version,
        "prompt_infos": prompt_infos,
        "data_typology": data_typology,
        "statistical_information": statistical_information,
        "source": "user",
    }


def generate_db_metadata_from_sdk(
    metadata,
    features,
    dataset_name_db,
    transformation=get_transformation_template("origin"),
    version="0.0.1",
    languages=["en"],
    data_typology="textdataset",
):

    summary = metadata["description"]
    homepage = metadata["homepage"]
    license = metadata["license"]
    languages = metadata["languages"]
    subset_name = metadata["config_name"]
    repository = metadata["repository"]
    leaderboard = metadata["leaderboard"]
    person_of_contact = metadata["person_of_contact"]
    huggingface_link = metadata["huggingface_link"]
    curation_rationale = metadata["curation_rationale"]
    genre = metadata["genre"]
    similar_datasets = metadata["similar_datasets"]
    creator_id = metadata["creator_id"]
    submitter_id = metadata["submitter_id"]
    multilinguality = metadata["multilinguality"]
    speaker_demographic = metadata["speaker_demographic"]
    annotator_demographic = metadata["annotator_demographic"]
    speech_situation = metadata["speech_situation"]
    prompt_infos = metadata["prompts"]

    task_categories = []  #
    tasks = []  #
    for value in metadata["task_templates"]:
        task_categories.append(value["task_categories"][0])
        tasks.append(value["task"])

    split = {}  #
    for key in metadata["splits"].keys():
        split[key] = metadata["splits"][key]["num_examples"]

    return validate_generate_db_metadata(
        dataset_name=dataset_name_db,
        transformation=transformation,
        version=version,
        task_categories=task_categories,
        tasks=tasks,
        split=split,
        languages=languages,
        summary=summary,
        homepage=homepage,
        license_=license,
        sub_dataset=subset_name,
        repository=repository,
        leaderboard=leaderboard,
        person_of_contact=person_of_contact,
        huggingface_link=huggingface_link,
        curation_rationale=curation_rationale,
        genre=genre,
        similar_datasets=similar_datasets,
        creator_name=creator_id,
        submitter_name=submitter_id,
        multilinguality=multilinguality,
        speaker_demographic=speaker_demographic,
        annotator_demographic=annotator_demographic,
        speech_situation=speech_situation,
        features=features,
        data_typology=data_typology,
        prompt_infos=prompt_infos,
    )
