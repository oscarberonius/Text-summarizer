from models.abstractive import Abstractive
from models.extractive import Extractive
from utils.data_processing import DataProcessor

from os.path import dirname
import sys

class Pipeline:

    def __init__(self, vscodearg=None, data_path='/data/en-ingress-dataset', query=None, save_path='/data/summary.txt'): 
        self.path = dirname(__file__)
        data_processor = DataProcessor()
        self.data = data_processor.process_data(self.path+data_path)
        self.query = query
        self.save_path = save_path
        self.extractive_summarizer = Extractive()
        self.abstractive_worder = Abstractive()

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