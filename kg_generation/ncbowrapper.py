import json
import os
import urllib.parse
import urllib.request

REST_URL = "http://data.bioontology.org"
REST_URL_BASE_ANNOTATOR_PARAMS = "/annotator?"
REST_URL_BASE_PROPERTY_SEARCH_PARAMS = "/property_search?"

SPARQL_ENDPOINT_URL = "http://sparql.bioontology.org"
SPARQL_BASE_ANNOTATOR_PARAMS = "/sparql?"

class NCBOWrapper:

    __key = None

    def __init__(self):
        self.__key = self.__getKey()

    def __getKey(self):
        return open("/mnt/clbp/ncbo_key.txt").read().strip()

    def annotate(self, contents, ontologies=None, max_level=0, include=None, longest_only=False):
        url = REST_URL + "/annotator"
        params = {
            "apikey": self.__key,
            "ontologies": ontologies,
            "longest_only": longest_only,
            "max_level": max_level,
            "include": "prefLabel,synonym,definition",
            #"text": urllib.parse.quote(contents)
            "text": contents
        }
        query_string = urllib.parse.urlencode( params )
        data = query_string.encode( "ascii" )

        #opener = urllib.request.build_opener()
        #opener.addheaders = [('Authorization', 'apikey token=' + self.__key)]
        #with opener.open( url, data ) as response:
        with urllib.request.urlopen( url, data) as response:
            response_text = response.read()
            #print( response_text )

        return json.loads(response_text)

    def annotate_get(self, contents, ontologies=None, max_level=0, include=None):
        params = REST_URL_BASE_ANNOTATOR_PARAMS
        if not ontologies is None:
            params += "ontologies=" + ontologies + "&" # ontologies must be a comma-separated list

        if max_level > 0:
            params += "max_level=" + max_level + "&"

        if include is None:
            include = "prefLabel,synonym,definition" # include must also be a comma-separated list
        params += "include=" + include + "&"

        params += "text=" + urllib.parse.quote(contents)
        url = REST_URL + params

        opener = urllib.request.build_opener()
        opener.addheaders = [('Authorization', 'apikey token=' + self.__key)]
        return json.loads(opener.open(url).read())

    def property_search(self, contents, ontologies=None):
        params = REST_URL_BASE_PROPERTY_SEARCH_PARAMS
        if not ontologies is None:
            params += "ontologies=" + ontologies + "&"

        params += "q=" + urllib.parse.quote(contents)
        url = REST_URL + params

        opener = urllib.request.build_opener()
        opener.addheaders = [('Authorization', 'apikey token=' + self.__key)]
        return json.loads(opener.open(url).read())

    def query(self, query_str=None):
        if query_str is None:
            query_str = 'SELECT ?s ?p ?o WHERE { ?s ?p ?o . } LIMIT 10'

        params = SPARQL_BASE_ANNOTATOR_PARAMS
        params += "apikey=" + self.__key + "&"
        params += "query=" + urllib.parse.quote(query_str) + "&"
        params += "outputformat=" + "json"

        url = SPARQL_ENDPOINT_URL + params
        opener = urllib.request.build_opener()
        return json.loads(opener.open(url).read())
