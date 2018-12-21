import json

class DataProcessor:
    def __init__(self):
        pass

    def process_data(self, path):
        text_ingress_list = self.get_text_ingress_list(path)
        print(text_ingress_list)

    def get_text_ingress_list(self, path):
        
        with open(path, 'r') as fp:
             obj = json.load(fp)

        quiddities = obj['views']['list']['results']
        text_ingress_list = []
        for quiddity in quiddities:
            text,ingress = self.get_text_ingress(quiddity['quiddity'])
            res = dict({'text': text, 'ingress': ingress})
            text_ingress_list.append(res)

        return text_ingress_list

    def get_text_ingress(self, dictionary):
        text = dictionary['body']['content']['text']
        ingress = dictionary['body']['ingress']['text']

        return text, ingress
