import unittest

from datalabs import load_dataset
from datalabs.operations.infer.inference import inference
from datalabs.operations.aggregate.auto_eval import explainaboard


@inference(name="my_inference")
def my_inference(samples: dict):
    predictions = []
    for sample in samples:
        text, label = sample["text"], sample["label"]
        predictions.append({"prediction": label})

    return predictions

class MyTestCase(unittest.TestCase):





    def test_general(self):

        # load dataset
        dataset = load_dataset("qc")
        # inference over test set based on a machine learning model
        test_data = dataset["test"].apply(my_inference, mode = "memory")
        # evaluation
        test_data = test_data.apply(explainaboard)

        # print the result
        print(test_data._stat)




if __name__ == '__main__':
    unittest.main()






