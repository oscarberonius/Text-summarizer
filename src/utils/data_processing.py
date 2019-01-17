import json
import tarfile
import os
import glob
from os.path import dirname
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

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
            # We use \t as start and \n as end symbols for target text
            res = dict({'text': text, 'ingress': '\t'+ingress+'\n'})
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
            # We use \t as start and \n as end symbols for target text
            res = dict({'text': text, 'ingress': '\t'+ingress+'\n'})
            text_ingress_list.append(res)

        return text_ingress_list
    
    # Retrieve text and ingress from dictionary
    def get_text_ingress(self, dictionary):
        text = dictionary['body']['content']['text']
        ingress = dictionary['body']['ingress']['text']

        return text, ingress

    def clean_data(self, data):
        path = dirname(__file__) + '/data'

        count = Counter()

        def baddict(dictionary, co):
            co[0] += 1
            if co[0] % 1000 == 0:
                print(co[0])
            for c in dictionary['text']:
                if ord(c) >= 256:
                    return True
            for c in dictionary['ingress']:
                if ord(c) >= 256:
                    return True
            return False

        clean_data = [d for d in data if not baddict(d, count)]

        with open('clean_data', 'w') as fout:
            json.dump(clean_data, fout)

    
    def visualize_data_point_sizes(self, dictionaries):
        x = []
        y = []
        for d in dictionaries:
            x.append(len(d['text']))
            y.append(len(d['ingress']))
        print("Mean length of texts: ", np.mean(x))
        print("Mean length of ingresses: ", np.mean(y))

        plt.subplot(211)
        plt.hist(x, range=(0, max(x)), color='red', label='text lengths')
        plt.subplot(212)
        plt.hist(y, range=(0, max(y)), color='blue', label='ingress lengths')
        plt.show()