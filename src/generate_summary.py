from models.abstractive import Abstractive
from models.extractive import Extractive
from utils.data_processing import DataProcessor

from os.path import dirname
import json
import sys
import tarfile

class Pipeline:

    def __init__(self, vscodearg=None, data_path='/data/en-ingress-dataset', query=None, save_path='/data/summary.txt'): 
        self.path = dirname(__file__)
        data_processor = DataProcessor()
        #self.data = data_processor.process_data(self.path+data_path)
        #data_processor.clean_data(self.data)
        self.query = query
        self.save_path = save_path

        tar1 = tarfile.open(self.path+'/data/clean_data1.tar.gz')
        info1 = tar1.getmembers()
        file1 = tar1.extractfile(info1[0])
        obj1 = json.load(file1)

        tar2 = tarfile.open(self.path + '/data/clean_data2.tar.gz')
        info2 = tar2.getmembers()
        file2 = tar2.extractfile(info2[0])
        obj2 = json.load(file2)

        self.data = obj1+obj2

        #data_processor.visualize_data_point_sizes(self.data)

        #self.extractive_summarizer = Extractive()
        self.abstractive_worder = Abstractive(self.data)

    def evaluate_cluster(self):
        extractive_summary = self.extractive_summarizer.summarize(self.data, self.query)
        self.abstractive_rewording = self.abstractive_worder.reword(extractive_summary)

    def generate_summary(self):
        self.format(self.abstractive_rewording)

    def format(self, summary):
        #Perform formatting if needed
        self.save(summary)

    def save(self, summary):
        path = self.path+self.save_path
        with open(path, 'w') as f:
            for word in summary:
                f.write(word+' ')

        print("Summary generated at ", path)

if __name__ == '__main__':
    args = sys.argv
    pipeline = Pipeline(args)
 #   pipeline.evaluate_cluster()
 #   pipeline.generate_summary()