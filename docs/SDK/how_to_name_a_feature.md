# How to name a feature:

#### A. For general features:

If there are two general features:  gender_bias_name_female &&  lexical_richness

If there are one split :   train

If there are two field:    text  && label

##### 1. data set level:

The name of feature should follow this format: 

{field name}\_{split_name}\_avg_{feature name}

Ex. text_train_avg_gender_bias_name_female

##### 2. sample level:

The name of feature should follow this format: 

{field name}_{feature name}

Ex. text_gender_bias_name_female

#### B.  ner

There are four features: true_entity_info_of  && avg_span_length_of (dataset level)  &&  avg_eCon_of (dataset level)  &&  avg_eFre_of (dataset level) 

If there are one split :   train

If there are two field:    tokens

##### 1. data set level:

The name of feature should follow this format: 

{feature name}\_{field name}_{split name}

Ex. avg_eFre_of_tokens_train

##### 2. sample level:

The name of feature should follow this format: 

{feature name}_{field name}

Ex. true_entity_info_of_tokens

#### C. nli

Usually, the field of nli dataset are premise and hypothesis

If there are one split :   train

There are three features: minus, add, divide

##### 1. data set level:

The name of feature should follow this format: 

premise\_length\_minus_hypothesis_avg\_{split name}_length

premise\_length\_add_hypothesis_avg\_{split name}_length

premise\_length\_divide_hypothesis_avg\_{split name}_length

Ex. premise\_length\_divide_hypothesis_avg\_train_length

##### 2. sample level:

The name of feature should follow this format: 

premise\_length\_minus_hypothesis_length

premise\_length\_add_hypothesis_length

premise\_length\_divide_hypothesis_length

#### D. QA

Usually, the field of nli dataset are question and context

If there are one split :   train

There are bleu features:bleu, divide

##### 1. data set level:

The name of feature should follow this format: 

question\_length\_divide_context\_avg\_{split name}_length

bleu_question_context_avg_{split name}

Ex. premise\_length\_divide_hypothesis_avg\_train_length

##### 2. sample level:

The name of feature should follow this format: 

question\_length\_divide_context_length

bleu_question_context

#### E. Summary

Usually, there are six features: density, coverage, compression, repetition, novelty, copy_length

If there are one split :   train

If the field of summary dataset are summary document

##### 1. data set level:

The name of feature should follow this format: 

```
avg_density_of_{split}_{field0}_and_{field1}
```

```
avg_coverage_of_{split}_{field0}_and_{field1}
```

```
avg_compression_of_{split}_{field0}_and_{field1}
```

```
avg_repetition_of_{split}_{field0}_and_{field1}
```

```
avg_novelty_of_{split}_{field0}_and_{field1}
```

```
avg_copy_length_of_{split}_{field0}_and_{field1}
```

Ex.  avg_copy_length_of\_train\_summary\_and_document

##### 2. sample level:

The name of feature should follow this format: 

```
density_of_{field0}_and_{field1}
```

```
coverage_of_{field0}_and_{field1}
```

```
compression_of_{field0}_and_{field1}
```

```
repetition_of_{field0}_and_{field1}
```

```
novelty_of_{field0}_and_{field1}
```

```
copy_length_of_{field0}_and_{field1}
```

Ex. copy_length_of\_summary\_and_document

