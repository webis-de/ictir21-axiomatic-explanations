import math
import time
import urllib.parse
import urllib.request

from axiomatic.axioms import prefs
from nltk.corpus import wordnet as wn
import itertools
from nltk.tokenize import RegexpTokenizer
import json
import requests



def argument_extractor_counter(tokdocument):
    _url = "http://ltdemos.informatik.uni-hamburg.de/arg-mining-ltcpu/classifyES_dep"
    _headers = {"accept": "application/json", "Content-Type": "text/plain"}

    argument_count = 0
    for item in tokdocument:
        r = requests.post(_url, headers=_headers, data=item.encode('utf-8'))
        # print(r.text)
        flag = 0
        for element in json.loads(r.text)[0]:
            # print(element['label'] + '\t' + element['token'])
            if flag == 0 and (element['label'] == 'C-B' or element['label'] == 'C-I'
                              or element['label'] == 'MC-B' or element['label'] == 'MC-I'):
                flag = 1
                argument_count += 1
            elif flag == 1 and (element['label'] == 'C-B' or element['label'] == 'C-I'
                                or element['label'] == 'MC-B' or element['label'] == 'MC-I'):
                pass
            elif flag == 1 and (element['label'] == 'P-B'):
                flag = 0
    return argument_count


# ------------------------------------------------------------------------


class Lookup(object):
    def __init__(self):
        self._cache = {}

    def _derive_key(self, params):
        return tuple(sorted(params.items()))

    def _raw_request(self, params={}, baseurl="", waittime=0):
        cache_key = self._derive_key(params)
        if cache_key not in self._cache:
            time.sleep(waittime)
            url_values = urllib.parse.urlencode(params)
            req = urllib.request.Request(baseurl + url_values, headers={'User-Agent': 'Mozilla'})

            handle = urllib.request.urlopen(req, timeout=120)
            encoding = handle.headers.get_content_charset()
            try:
                content = str(handle.read().decode("latin-1"))
            except:
                content = str(handle.read().decode(encoding, errors='ignore'))

            self._cache[cache_key] = ' '.join(content.split())
        return self._cache[cache_key]


_CHATNOIR2APIKEY = "97fdd7c0-49ae-4d30-ad5a-7953948cdad9"


class Chatnoir1Clueweb09IdfLookup(Lookup):
    _baseurl = "http://betaweb135:8081/by_key?"

    def _derive_key(self, params):
        return params['key']

    def get(self, term):
        value = json.loads(self._raw_request({"key": term}, self._baseurl))
        if value["key"] != "invalid":
            return float(value["value"])
        return 0.0


class Chatnoir1Clueweb09PagerankLookup(Lookup):
    _baseurl = "http://betaweb135:8080/by_key?"

    def _derive_key(self, params):
        return params['key']

    def get(self, docid):
        value = json.loads(self._raw_request({"key": docid}, self._baseurl))
        if value["key"] != "invalid":
            return float(value["value"])
        return 0.0


_idfLookup = None
_pagerankLookup = None


def set_idf_lookup(lookup):
    global _idfLookup
    _idfLookup = lookup


def set_pagerank_lookup(lookup):
    global _pagerankLookup
    _pagerankLookup = lookup


def get_idf_value(term):
    return _idfLookup.get(term)


def get_pagerank_value(docid):
    return _pagerankLookup.get(docid)

from axiomatic.features.computed import Chatnoir1Clueweb09IdfLookup, Chatnoir1Clueweb09PagerankLookup, \
    _idfLookup, _pagerankLookup, set_idf_lookup, set_pagerank_lookup


# ------------------------------------------------------------------------


if _idfLookup is None:
    set_idf_lookup(Chatnoir1Clueweb09IdfLookup())

if _pagerankLookup is None:
    set_pagerank_lookup(Chatnoir1Clueweb09PagerankLookup())
