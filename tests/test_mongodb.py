import unittest
import datasets

class MyTestCase(unittest.TestCase):
    def test_load_new_dataset(self):
        #dataset = datasets.load_dataset("datasets/adv_mtl")
        dataset = datasets.load_dataset("adv_mtl")
        self.assertEqual(len(dataset["imdb_test"]["sentence"]), 400)

        # write it to mongodb
        #dataset["imdb_test"].write_db()

        # drop the collection of mongodb
        # cluster = datasets.MongoDBClient("cluster0")
        # cluster.drop("dev_samples_of_dataset", "adv_mtl", True)


