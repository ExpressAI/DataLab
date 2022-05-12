import unittest

from datalabs.tasks import get_task, TaskType


class MyTestCase(unittest.TestCase):
    def test_something(self):

        print(TaskType.text_pair_classification)
        print(get_task(TaskType.text_pair_classification)(text1_column="aha"))


if __name__ == "__main__":
    unittest.main()
