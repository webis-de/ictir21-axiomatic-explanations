import unittest
import json
from approvaltests import verify_as_json


class ReconstructOriginalIdsTest(unittest.TestCase):
    def system_ranks(self, tmp):
        ret = {}
        for i in tmp:
            i = json.loads(i)

            for j in i:
                if str(j['system']) not in ret:
                    ret[str(j['system'])] = {}
                if not str(j['query']) in ret[j['system']]:
                    ret[j['system']][str(j['query'])] = {}

                ret[str(j['system'])][str(j['query'])][str(j['rank'])] = (j['docid'], j['score'])

        return ret

    def qrel_preference(self, first_rel, second_rel):
        if first_rel is None or second_rel is None:
            return None

        return 'correct' if first_rel >= second_rel else 'wrong'

    def parse_bm25_run(self, file_path):
        with open(file_path, 'r') as f:
            ret = {}
            for l in f:
                topic = l.split()[0]
                doc = l.split()[2]
                pos = l.split()[3]

                if topic not in ret:
                    ret[topic] = {}
                ret[topic][doc] = pos

            return ret

    def parse_relevance(self, file_path):
        with open(file_path, 'r') as f:
            ret = {}
            for l in f:
                topic = l.split()[0]
                doc = l.split()[2]
                rel = l.split()[3]

                if topic not in ret:
                    ret[topic] = {}
                ret[topic][doc] = rel

            return ret

    def map(self, system_ranks, relevance, bm25_run, pair):
        ret = json.loads(pair)
        if ret['concordant'] == 1:
            ret['id1'] = system_ranks[str(ret['system'])][str(ret['query'])][str(ret['upper_rank'] - ret['rankdiff'])][0]
            ret['id2'] = system_ranks[str(ret['system'])][str(ret['query'])][str(ret['upper_rank'])][0]
            ret['id1_score'] = system_ranks[str(ret['system'])][str(ret['query'])][str(ret['upper_rank'] - ret['rankdiff'])][1]
            ret['id2_score'] = system_ranks[str(ret['system'])][str(ret['query'])][str(ret['upper_rank'])][1]
            ret['id1_relevance'] = relevance.get(str(ret['query']), {}).get(str(ret['id1']), None)
            ret['id2_relevance'] = relevance.get(str(ret['query']), {}).get(str(ret['id2']), None)

            ret['id1_pos_in_bm25'] = bm25_run.get(str(ret['query']), {}).get(str(ret['id1']), None)
            ret['id2_pos_in_bm25'] = bm25_run.get(str(ret['query']), {}).get(str(ret['id2']), None)

            ret['qrel_preference'] = self.qrel_preference(ret['id1_relevance'], ret['id2_relevance'])
            ret['bm25_preference_opposite'] = ret['id1_pos_in_bm25'] is not None \
                                                and ret['id2_pos_in_bm25'] is not None \
                                                and ret['id1_pos_in_bm25'] > ret['id2_pos_in_bm25']

            return ret

    def bla(self, system_ranks, relevance, bm25_run, pairs):
        return [j for j in [self.map(system_ranks, relevance, bm25_run, i) for i in pairs] if j is not None]

    def test_reconstruction_with_single_input_line(self):
        run_file = ['''[
                {"system":"s1", "query":690, "docid": "FT944-12903", "rank": 1, "score": 10.0},
                {"system":"s1", "query":690, "docid": "LA110789-0114", "rank": 2, "score": 9.0},
                {"system":"s1", "query":690, "docid": "FT931-16546", "rank": 3, "score": 8.0}]''']
        reduce_output = open('test/BuildTrainingSetTest.test_reducer_on_map_output_with_single_input_line.approved.txt')\
            .read().split('\n')

        verify_as_json(self.bla(
            self.system_ranks(run_file),
            self.parse_relevance('test/test-qrels.txt'),
            self.parse_bm25_run('test/test-run.txt'),
            reduce_output)
        )

    def test_reconstruction_with_multiple_input_lines(self):
        run_file = ['''[
                {"system":"s1", "query":690, "docid": "FT944-12903", "rank": 1, "score": 1.0},
                {"system":"s1", "query":690, "docid": "LA110789-0114", "rank": 2, "score": 0.9},
                {"system":"s1", "query":690, "docid": "FT931-16546", "rank": 3, "score": 0.8}]''',
                    '''[
                {"system":"s2", "query":312, "docid": "LA040289-0050", "rank": 1, "score": 1.0},
                {"system":"s2", "query":312, "docid": "FR941007-0-00016", "rank": 2, "score": 0.9}]''']
        reduce_output = open('test/BuildTrainingSetTest.test_reducer_on_map_output_with_multiple_input_lines.approved.txt')\
            .read().split('\n')

        verify_as_json(self.bla(
            self.system_ranks(run_file),
            self.parse_relevance('test/test-qrels.txt'),
            self.parse_bm25_run('test/test-run.txt'),
            reduce_output
        ))
