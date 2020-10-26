import unittest
from approvaltests import verify_as_json
from axiomatic.explanations.experiment_lib import generate_pairs_from_qrels


class TestGenerationOfPairsFromJudgments(unittest.TestCase):
    def test_generation_for_small_sample_qrels(self):
        actual = generate_pairs_from_qrels('test/sample-small-qrels.txt')

        verify_as_json([i for i in actual])

    def test_generation_for_medium_sample_qrels(self):
        actual = generate_pairs_from_qrels('test/sample-medium-qrels.txt')

        verify_as_json([i for i in actual])

    def test_generation_for_large_sample_qrels(self):
        actual = generate_pairs_from_qrels('test/sample-large-qrels.txt')

        verify_as_json([i for i in actual])
