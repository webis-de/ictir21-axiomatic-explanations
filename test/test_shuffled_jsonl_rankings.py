import unittest
from approvaltests import verify
from axiomatic.collection.anserini import shuffled_jsonl_content


class TestShuffledJsonlRankings(unittest.TestCase):

    def test_shuffling_of_sample_run_file(self):
        actual = shuffled_jsonl_content(
            'test/TestRunFIleTransformation.test_transformation_of_sample_run_file.approved.txt'
        )
        verify(actual)

    def test_shuffling_of_sample_run_file_with_multiple_queries(self):
        actual = shuffled_jsonl_content(
            'test/TestRunFIleTransformation.test_transformation_of_sample_run_file_with_multiple_queries.approved.txt'
        )
        verify(actual)
