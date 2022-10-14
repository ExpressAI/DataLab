# Schema of Report

## Text Classification

```python
    res = {
            "dataset-level":{
                "length_info": {
                    "max_text_length": np.max(lengths),
                    "min_text_length": np.min(lengths),
                    "average_text_length": np.average(lengths),
                },
                "label_info": {
                    "ratio":min(labels_to_number.values()) * 1.0 / max(labels_to_number.values()),
                    "distribution": labels_to_number,
                },
                "gender_info":gender_ratio,
                # "vocabulary_info":vocab_sorted,
                "number_of_samples":len(samples),
                "number_of_tokens":number_of_tokens,
                "hatespeech_info":hatespeech,
                "spelling_errors":len(spelling_errors),
            },
        "sample-level":sample_infos
    }
```

![image](https://user-images.githubusercontent.com/59123869/149849695-615304ff-a19f-4dca-8597-c1a7bca41363.png)

## Text Pair Classification

```Python
    res = {
            "dataset-level":{
                "length_info": {
                    "max_text1_length": np.max(text1_lengths),
                    "min_text1_length": np.min(text1_lengths),
                    "average_text1_length": np.average(text1_lengths),
                    "max_text2_length": np.max(text2_lengths),
                    "min_text2_length": np.min(text2_lengths),
                    "average_text2_length": np.average(text2_lengths),
                    "text1_divided_text2":np.average(text1_divided_text2),
                },
                "label_info": {
                    "ratio": min(labels_to_number.values()) * 1.0 / max(labels_to_number.values()),
                    "distribution": labels_to_number,
                },
                "vocabulary_info":vocab_sorted,
                "number_of_samples": len(samples),
                "number_of_tokens": number_of_tokens,
                "gender_info": gender_ratio,
                "average_similarity": np.average(similarities),
                "hatespeech_info": hatespeech,
            },
        "sample-level": sample_infos
    }
```

![image](https://user-images.githubusercontent.com/59123869/149858573-1b494181-1e1d-4ec1-9043-4c7102303a9f.png)

## Named Entity Recognition

```python
    res = {
        "dataset-level": {
            "entity_info":{
                "avg_entity_length": avg_entityLen,
                "avg_entity_on_sentence": avg_entity_nums_inSent,
                "sentence_without_entity": len(samples) - len(chunks),
                "entity_length_distribution": entity_length_distribution,
            },
            "length_info": {
                "max_text_length": np.max(lengths),
                "min_text_length": np.min(lengths),
                "average_text_length": np.average(lengths),
            },
            "label_info": {
                "ratio": min(labels_to_number.values()) * 1.0 / max(labels_to_number.values()),
                "distribution": label_distribution, #labels_to_number,
            },
            "gender_info": gender_ratio,
            "vocabulary_info":vocab_sorted,
            "number_of_samples": len(samples),
            "number_of_tokens": number_of_tokens,
            "hatespeech_info": hatespeech,
        },
        "sample-level": sample_infos
    }
```

![截屏2022-01-18 下午4 20 06](https://user-images.githubusercontent.com/24972331/149898182-96462875-afd4-4edf-9a95-bf5407abe10d.png)

## Summarization

```python
    res = {
        "dataset-level":{
                "average_text_length":np.average(text_lengths),
                "average_summary_length":np.average(summary_lengths),
                "length_info": {
                    "max_text_length": np.max(text_lengths),
                    "min_text_length": np.min(text_lengths),
                    "average_text_length": np.average(text_lengths),
                    "max_summary_length": np.max(summary_lengths),
                    "min_summary_length": np.min(summary_lengths),
                    "average_summary_length": np.average(summary_lengths),
                },
                "number_of_samples": len(samples),
                "number_of_tokens": number_of_tokens,
                "vocabulary_info": vocab_sorted,
                "gender_info": gender_ratio,
                "hatespeech_info": hatespeech,
                **attr_avg,
        },
        "sample-level": sample_infos,
    }
```
