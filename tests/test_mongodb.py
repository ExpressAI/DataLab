import unittest
import datasets

class MyTestCase(unittest.TestCase):
    def test_load_new_dataset(self):
        #dataset = datalab.load_dataset("datalab/adv_mtl")
        dataset = datasets.load_dataset("adv_mtl")
        self.assertEqual(len(dataset["imdb_test"]["sentence"]), 400)

        # write it to mongodb
        #dataset["imdb_test"].write_db()

        # drop the collection of mongodb
        # cluster = datalab.MongoDBClient("cluster0")
        # cluster.drop("dev_samples_of_dataset", "adv_mtl", True)


