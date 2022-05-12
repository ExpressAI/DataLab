import unittest

from datalabs.tasks.task_registry import get_task, TASK_REGISTRY


class MyTestCase(unittest.TestCase):
    def test_something(self):

        print(get_task("summarization")())
        # print(TASK_REGISTRY)

        for task_name in TASK_REGISTRY.keys():
            print(task_name)


if __name__ == "__main__":
    unittest.main()
