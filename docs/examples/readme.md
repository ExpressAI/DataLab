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
