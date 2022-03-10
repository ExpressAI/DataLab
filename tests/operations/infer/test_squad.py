import unittest
from datalabs import load_dataset
from datalabs.operations.infer.inference import inference
from datalabs.operations.aggregate.auto_eval import explainaboard


@inference(name="my_inference")
def my_inference(samples: list):
    predictions = []
    for sample in samples:
        context, question, answers = sample["context"], sample["question"], sample["answers"]


        prediction:str = "dummy_answer" # this should be modified based on user's model

        predictions.append({"prediction": prediction})

    return predictions


class MyTestCase(unittest.TestCase):
    def test_general(self):
        # 1. load dataset
        dataset = load_dataset("squad") # subdataset (e.g., en or zh) -> dataset (en + zh)

        # 2. inference over test set based on a machine learning model
        test_data = dataset["validation"].apply(my_inference, mode="memory")

        # 3. evaluation
        #explainaboard.resources = {"dataset_info":dataset["test"]._info}
        test_data = test_data.apply(explainaboard)

        # 4. print the result
        print(test_data._stat)


if __name__ == '__main__':
    unittest.main()
