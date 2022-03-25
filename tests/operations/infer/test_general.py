import unittest
from datalabs import load_dataset
from datalabs.operations.infer.inference import inference
# from datalabs.operations.aggregate.auto_eval import explainaboard
from explainaboard.utils.eval_ops import explainaboard

@inference(name="my_inference")
def my_inference(samples: list):
    """
    This is an example using Huggingface Pipeline to do sentiment classification
    One thing we need to notice is that in Datalab, positive is 0, negative is 1
    """
    from transformers import pipeline
    classifier = pipeline('sentiment-analysis')
    examples = [x["text"] for x in samples]
    outs = classifier(examples)
    predictions = []
    for out in outs:
        if out["label"] == "POSITIVE":
            # In DataLab, positive is 0, negative is 1
            predictions.append({"prediction": 0})
        else:
            predictions.append({"prediction": 1})
    return predictions


class MyTestCase(unittest.TestCase):
    def test_general(self):
        # load dataset
        dataset = load_dataset("sst2")
        # inference over test set based on a machine learning model
        test_data = dataset["test"].apply(my_inference, mode="memory")
        # evaluation
        test_data = test_data.apply(explainaboard)
        # print the result
        #print(test_data._stat)


if __name__ == '__main__':
    unittest.main()
