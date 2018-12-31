import json
import tarfile
import os

class DataProcessor:
    def __init__(self):
        pass

    def process_data(self, path):
        if os.path.isfile(path):
            text_ingress_list = self.get_text_ingress_list(path)
        else:
            text_ingress_list = self.read_data_folder(path)

    def read_data_folder(self, path):  
        
        tar = tarfile.open(path+'/02d8cf54-a762-46a8-9251-944f3d43d0b2_00001_processed.tar.gz')
        text_ingress_list = self.process_messages(tar)
        tar.close()
        # Untar all tarred folders in path

    def process_messages(self, tar):
        text_ingress_list = []

        # Extract messages and retrieve texts and ingresses as dicts.
        for member in tar.getmembers():
            fileobj = tar.extractfile(member)
            message = json.load(fileobj)
            text, ingress = self.get_text_ingress(message)
            res = dict({'text': text, 'ingress': ingress})
            text_ingress_list.append(res)

        return text_ingress_list
                
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
