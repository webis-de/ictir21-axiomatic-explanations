import unittest
import tempfile
from approvaltests import verify


class TestRunFIleTransformation(unittest.TestCase):
    def test_transformation_of_sample_run_file(self):
        actual_transformed_content = self.transform_run_file('test/sample_run_file.txt')
        verify(actual_transformed_content)

    def test_transformation_of_sample_run_file_with_multiple_queries(self):
        actual_transformed_content = self.transform_run_file('test/sample_run_file_multiple_queries.txt')
        verify(actual_transformed_content)

    @staticmethod
    def transform_run_file(file_name):
        from axiomatic.collection.anserini import run_file_to_jsonl
        with tempfile.NamedTemporaryFile() as tmp_file:
            run_file_to_jsonl(input_file=file_name, output_file=tmp_file.name)
            return open(tmp_file.name).read()
