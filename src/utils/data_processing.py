import json

class DataProcessor:
    def __init__(self):
        pass

    def process_data(self, path):
        quiddities = self.process_search_result(path)
        text,ingress = self.get_text_ingress(quiddities[0]['quiddity'])

    def process_search_result(self, path):
        with open(path, 'r') as fp:
             obj = json.load(fp)
        
        quiddities = obj['views']['list']['results']

        return quiddities

    def get_text_ingress(self, dictionary):
        text = dictionary['body']['content']['text']
        ingress = dictionary['body']['ingress']['text']

        return text, ingress
