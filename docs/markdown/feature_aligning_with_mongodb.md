
## `datasets.Dataset`

### `write_arrow(path: str)`
将当前数据集的样本以 .arrow 文件的形式写出到指定路径。
- 参数
    - `path (str)`：写出路径，直到写出文件名，例如 `../out.arrow`。

### `write_db()`
将当前数据集的元数据（`DatasetInfo`）和所有样本上传到云端 MongoDB。

## `datasets.DatasetInfo`
- 属性
    - `description (str)`：不变。数据集的描述；
    - `citation (str)`: 不变。数据集的 BibTex 引用；
    - `homepage (str)`：不变。该数据集的官方主页 URL；
    - `license (str)`：不变。数据集的证书，可以是证书名或证书内容；
    - `features (Features)`：不变，可选。数据集的列特征；
    - `post_processed (PostProcessedInfo)`：不变，可选。有关数据集可能的处理后资源的信息；
    - `supervised_keys (SupervisedKeysData)`: 不变，可选。如果适用于数据集，则指定监督学习的输入特征和标签；
    - `task_templates (List[TaskTemplate])`：不变，可选。在训练和评估期间准备数据集的任务模板；
    - `builder_name (str)`：不变，可选。用于创建数据集的 `GeneratorBasedBuilder` 子类的名称，通常匹配到相应的脚本名称，是数据集构建器类名的蛇形命名版；
    - `config_name (str)`：不变，可选。从 `BuilderConfig` 得到的配置名；
    - `version (str or Version)`：不变，可选。数据集版本号；
    - `splits (dict)`：不变，可选。名称和元数据之间的映射；
    - `download_checksums (dict)`：不变，可选。下载数据集校验和的 URL 与相应元数据之间的映射；
    - `download_size (int)`：被隐藏，可选。
    - `post_processing_size (int)`：被隐藏，可选。
    - `dataset_size (int)`：被隐藏，可选。
    - `size_in_bytes (int)`：被隐藏，可选。
    - `dataset_name (str)`：新增加，可选。
    - `sub_dataset (str)`：新增加，可选。
    - `homepage (str)`：新增加，可选。
    - `repository (str)`：新增加，可选。
    - `leaderboard (str)`：新增加，可选。
    - `person_of_contact (str)`：新增加，可选。
    - `production_status (str)`：新增加，可选。
    - `huggingface_link (str)`：新增加，可选。
    - `curation_rationale (str)`：新增加，可选。
    - `genre (str)`：新增加，可选。
    - `quality (str)`：新增加，可选。
    - `similar_datasets (str)`：新增加，可选。
    - `creator_id (str)`：新增加，可选。
    - `submitter_id (str)`：新增加，可选。
    - `Multilinguality (str)`：新增加，可选。
    - `transformation (str)`：新增加，可选。
    - `languages (List[str])`：新增加，可选。
    - `model_ids (List)`：新增加，可选。
    - `speaker_demographic (SpeakerDemographic)`：新增加，可选。
    - `annotator_demographic (AnnotatorDemographic)`：新增加，可选。
    - `speech_situation (SpeechSituation)`：新增加，可选。
    - `size (SizeInfo)`：新增加，可选。
    - `popularity (Popularity)`：新增加，可选。

### `_init_db_attr()`
内建方法，请不要在 `DatasetInfo` 类外调用。此方法尝试从 `MongoDB` 获得同名数据集元数据，并填充到该类的属性中。

### `_infer_attr()`
内建方法，请不要在 `DatasetInfo` 类外调用。此方法从尝试从已有的基本信息中推断出未获得的元数据的可能值。

### `_fill_single_attr(src: dict, attr: str)`
内建方法，请不要在 `DatasetInfo` 类外调用。此方法将指定的键值对值转移到自己的属性中。

### `_fill_db_attr()`
内建方法，请不要在 `DatasetInfo` 类外调用。此方法将所有新定义的键值对转移到自己的属性中。

### `_as_dict()`
内建方法，请不要在 `Dataset` 类外调用。此方法将数据集信息转化为字典。


## `SpeakerDemographic`
内建类，请不要在`DatasetInfo` 类外使用。此类是 `DatasetInfo` 类中 `speaker_demographic` 属性对应的数据类，在 `DatasetInfo` 转换为字典时会被递归地转化为字典。
- 属性
    - `gender (str)`：可选。
    - `race (str)`：可选。
    - `ethnicity (str)`：可选。
    - `native_language (str)`：可选。
    - `socioeconomic_status (str)`：可选。
    - `number_of_different_speakers_represented (str)`：可选。
    - `presence_of_disordered_speech (str)`：可选。
    - `training_in_linguistics (str)`：可选。

## `AnnotatorDemographic`
内建类，请不要在`DatasetInfo` 类外使用。此类是 `DatasetInfo` 类中 `annotator_demographic` 属性对应的数据类，在 `DatasetInfo` 转换为字典时会被递归地转化为字典。
- 属性
    - `gender (str)`：可选。
    - `race (str)`：可选。
    - `ethnicity (str)`：可选。
    - `native_language (str)`：可选。
    - `socioeconomic_status (str)`：可选。
    - `number_of_different_speakers_represented (str)`：可选。
    - `presence_of_disordered_speech (str)`：可选。
    - `training_in_linguistics (str)`：可选。

## `SpeechSituation`
内建类，请不要在`DatasetInfo` 类外使用。此类是 `DatasetInfo` 类中 `speech_situation` 属性对应的数据类，在 `DatasetInfo` 转换为字典时会被递归地转化为字典。
- 属性
    - `time (str)`：可选。
    - `place (str)`：可选。
    - `modality (str)`：可选。
    - `intended_audience (str)`：可选。

## `SizeInfo`
内建类，请不要在`DatasetInfo` 类外使用。此类是 `DatasetInfo` 类中 `size` 属性对应的数据类，在 `DatasetInfo` 转换为字典时会被递归地转化为字典。
- 属性
    - `samples (int)`：可选。
    - `storage (str)`：可选。

## `Popularity`
内建类，请不要在`DatasetInfo` 类外使用。此类是 `DatasetInfo` 类中 `popularity` 属性对应的数据类，在 `DatasetInfo` 转换为字典时会被递归地转化为字典。
- 属性
    - `number_of_download (int)`：可选。
    - `number_of_times (int)`：可选。
    - `number_of_reposts (int)`：可选。
    - `number_of_visits (int)`：可选。
