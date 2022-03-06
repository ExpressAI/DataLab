import unittest

from datalabs import operations, load_dataset
from datalabs.operations.infer.inference import inference


@inference(name="my_inference")
def my_inference(samples: dict):
    predictions = []
    for sample in samples:
        text, label = sample["text"], sample["label"]
        predictions.append({"prediction": label})

    return predictions

class MyTestCase(unittest.TestCase):





    def test_general(self):

        dataset = load_dataset("qc")

        res = dataset["test"].apply(my_inference, mode = "local")
        print(res)




if __name__ == '__main__':
    unittest.main()






