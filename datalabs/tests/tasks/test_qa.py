import unittest

from datalabs.tasks.question_answering import (
    QuestionAnswering,
    QuestionAnsweringAbstractive,
    QuestionAnsweringAbstractiveNQ,
    QuestionAnsweringDCQA,
    QuestionAnsweringExtractive,
    QuestionAnsweringHotpot,
    QuestionAnsweringMultipleChoice,
    QuestionAnsweringMultipleChoiceQASC,
    QuestionAnsweringMultipleChoiceWithoutContext,
)


class MyTestCase(unittest.TestCase):
    def test_something(self):

        print(QuestionAnswering())
        print(QuestionAnsweringAbstractive())
        print(QuestionAnsweringAbstractiveNQ())
        print(QuestionAnsweringDCQA())
        print(QuestionAnsweringExtractive())
        print(QuestionAnsweringHotpot())
        print(QuestionAnsweringMultipleChoice())
        print(QuestionAnsweringMultipleChoiceQASC())
        print(QuestionAnsweringMultipleChoiceWithoutContext())


if __name__ == "__main__":
    unittest.main()
