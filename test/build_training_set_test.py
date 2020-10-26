import unittest
import os
from io import StringIO
from experiments.build_training_set_mapred import main
from approvaltests import verify
from unittest.mock import patch


class BuildTrainingSetTest(unittest.TestCase):
    ENV_DIR = '.'

    @classmethod
    def setUpClass(cls):
        os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'

    def test_map_with_invalid_json_input_lines(self):
        expected_output = ''
        actual_output = self.run_map_job(
            json_input_lines=['', '{as', '{a}']
        )
        self.assertEquals(expected_output, actual_output)

    def test_map_with_single_input_line(self):
        actual_output = self.run_map_job(
            json_input_lines=['''[
                {"system":"s1", "query":690, "docid": "FT944-12903", "rank": 1, "score": 10.0},
                {"system":"s1", "query":690, "docid": "LA110789-0114", "rank": 2, "score": 9.0},
                {"system":"s1", "query":690, "docid": "FT931-16546", "rank": 3, "score": 8.0}]''']
        )

        verify(actual_output)

    def test_map_with_multiple_input_lines(self):
        actual_output = self.run_map_job(
            json_input_lines=['''[
                {"system":"s1", "query":690, "docid": "FT944-12903", "rank": 1, "score": 1.0},
                {"system":"s1", "query":690, "docid": "LA110789-0114", "rank": 2, "score": 0.9},
                {"system":"s1", "query":690, "docid": "FT931-16546", "rank": 3, "score": 0.8}]''',
                              '''[
                {"system":"s2", "query":312, "docid": "LA040289-0050", "rank": 1, "score": 1.0},
                {"system":"s2", "query":312, "docid": "FR941007-0-00016", "rank": 2, "score": 0.9}]''']
        )

        verify(actual_output)

    def test_reducer_on_map_output_with_single_input_line(self):
        map_output = open('test/BuildTrainingSetTest.test_map_with_single_input_line.approved.txt').read().split('\n')
        actual_output = self.run_reduce_job_on_robust(map_output)

        verify(actual_output)

    def test_reducer_on_map_output_with_multiple_input_lines(self):
        map_output = open('test/BuildTrainingSetTest.test_map_with_multiple_input_lines.approved.txt').read().split('\n')
        actual_output = self.run_reduce_job_on_robust(map_output)

        verify(actual_output)

    def test_map_with_multiple_input_lines_for_ms_marco(self):
        actual_output = self.run_map_job(
            json_input_lines=['''[
                {"system":"s1", "query":1136427, "docid": "D1345527", "rank": 1, "score": 1.0},
                {"system":"s1", "query":1136427, "docid": "D1822834", "rank": 2, "score": 0.9},
                {"system":"s1", "query":1136427, "docid": "D183343", "rank": 3, "score": 0.8}]''',
                              '''[
                {"system":"s2", "query":11096, "docid": "D440477", "rank": 1, "score": 1.0},
                {"system":"s2", "query":11096, "docid": "D3172928", "rank": 2, "score": 0.9}]''']
        )

        verify(actual_output)

    def test_reducer_on_map_output_with_multiple_input_lines_for_ms_marco(self):
        map_output = open('test/BuildTrainingSetTest.test_map_with_multiple_input_lines_for_ms_marco.approved.txt')\
            .read().split('\n')
        actual_output = self.run_reduce_job_on_ms_marco(map_output)

        verify(actual_output)

    def run_map_job(self, json_input_lines):
        os.environ['DATASET'] = 'robust04'
        with patch('sys.stdout', new=StringIO()) as stdout, \
                patch('sys.stdin', new=json_input_lines), \
                patch.dict(os.environ, {'TOPICS_FILE': 'topics.robust04.301-450.601-700.txt'}):
            main(
                mode='ranking2pairs',
                env_dir=self.ENV_DIR
            )

            return self.clean_output(stdout)

    def run_reduce_job_on_robust(self, map_output):
        os.environ['DATASET'] = 'robust04'
        with patch('sys.stdout', new=StringIO()) as stdout, \
                patch('sys.stdin', new=map_output), \
                patch.dict(os.environ, {'TOPICS_FILE': 'topics.robust04.301-450.601-700.txt'}):
            main(
                mode='pairs2prefs',
                env_dir=self.ENV_DIR
            )

            ret = stdout.getvalue().strip().split('\n')
            ret = sorted(ret)

            return '\n'.join(ret)

    def run_reduce_job_on_ms_marco(self, map_output):
        os.environ['DATASET'] = 'ms-marco'
        with patch('sys.stdout', new=StringIO()) as stdout, \
                patch('sys.stdin', new=map_output), \
                patch.dict(os.environ, {'TOPICS_FILE': 'msmarco-test2019-queries.tsv'}):
            main(
                mode='pairs2prefs',
                env_dir=self.ENV_DIR
            )

            ret = stdout.getvalue().strip().split('\n')
            ret = sorted(ret)

            return '\n'.join(ret)

    @staticmethod
    def clean_output(output):
        ret = []
        for line in output.getvalue().strip().split('\n'):
            if '\t' in line:
                line = "<HASH_REMOVED>\t" + line.split('\t')[1]
            ret += [line]

        return '\n'.join(ret)
