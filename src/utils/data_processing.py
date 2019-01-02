import json
import tarfile
import os
import glob

class DataProcessor:
    def __init__(self):
        pass

    # Process given data files, return list(dict({text, ingress}))
    def process_data(self, path):
        if os.path.isfile(path):
            text_ingress_list = self.get_text_ingress_list(path)
        else:
            text_ingress_list = self.read_data_folder(path)

        return text_ingress_list

    # Process data that is on the format folder/[tar.gz(folder)]/[message]
    def read_data_folder(self, path):
        text_ingress_list = []

        # Open all tar files and process the containing messages
        for f in glob.glob(os.path.join(path, '*.tar.gz')):
            tar = tarfile.open(f)
            current_text_ingress_list = self.process_messages(tar)
            text_ingress_list.extend(current_text_ingress_list)
            tar.close()

        return text_ingress_list

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

    # Process data that is on the format of a file containing quiddities            
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
    
    # Retrieve text and ingress from dictionary
    def get_text_ingress(self, dictionary):
        text = dictionary['body']['content']['text']
        ingress = dictionary['body']['ingress']['text']

        return text, ingress
