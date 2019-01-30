from __future__ import division
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
            res = dict({'text': text, 'ingress': '\t' + ingress + '\n'})
            text_ingress_list.append(res)

        return text_ingress_list

    # Process data that is on the format of a file containing quiddities            
    def get_text_ingress_list(self, path):
        with open(path, 'r') as fp:
            obj = json.load(fp)

        quiddities = obj['views']['list']['results']
        text_ingress_list = []
        for quiddity in quiddities:
            text, ingress = self.get_text_ingress(quiddity['quiddity'])
            # We use \t as start and \n as end symbols for target text
            res = dict({'text': text, 'ingress': '\t' + ingress + '\n'})
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
        z = []
        for d in dictionaries:
            x.append(len(d['text']))
            y.append(len(d['ingress']))
            if len(d['text']) > 0:
                val = len(d['ingress']) / len(d['text'])
                z.append(val)
        print("Max length of texts = {mean_text}.".format(mean_text=np.max(x)))
        print("Max length of ingresses = {mean_text}.".format(mean_text=np.max(y)))
        print("Max size ratio ingress/text = {mean_text}.".format(mean_text=np.max(z)))
        print("Median length of texts = {mean_text}.".format(mean_text=np.median(x)))
        print("Median length of ingresses = {mean_ingress}.".format(mean_ingress=np.median(y)))
        print("Median size ratio ingress/text = {mean_ratio}.".format(mean_ratio=np.median(z)))
        print("Average length of text= {avg_ingress}.".format(avg_ingress=np.average(x)))
        print("Average length of ingresses = {avg_ingress}.".format(avg_ingress=np.average(y)))
        print("Average size ratio ingress/text = {avg_ingress}.".format(avg_ingress=np.average(z)))
        print("Data points: {data_points}".format(data_points=len(dictionaries)))

        plt.subplot(311)
        plt.hist(x, range=(0, max(x)), color='red', label='text lengths')
        plt.subplot(312)
        plt.hist(y, range=(0, max(y)), color='blue', label='ingress lengths')
        plt.subplot(313)
        plt.hist(z, range=(0, max(z)), color='green', label='ingress/text ratios')
        plt.show()

    def remove_large_texts(self, dictionaries, text_threshold, ingress_threshold):
        reduced = [d for d in dictionaries if
                   len(d['text']) <= text_threshold and len(d['ingress']) <= ingress_threshold]
        return reduced

    def remove_unbalanced_texts(self, dictionaries, low_end, high_end):
        # Removes all dictionaries where length(ingress/length(text) is outside low_end - high_end (including empty elements)
        reduced = [d for d in dictionaries if
                   len(d['ingress']) > 0 and len(d['text']) > 0 and low_end <= len(d['ingress']) / len(
                       d['text']) <= high_end]
        return reduced
