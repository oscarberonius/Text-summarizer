from models.abstractive import Abstractive
from models.extractive import Extractive
from utils.data_processing import DataProcessor

import sys

class Pipeline:

    def __init__(self, data_path='../data/cluster', query=None, save_path='../data/summary.txt'): 
        data_processor = DataProcessor()
        self.data = data_processor.process_data(data_path)
        self.query = query
        self.save_path = save_path
        self.extractive_summarizer = Extractive()
        self.abstractive_worder = Abstractive()

    def evaluate_cluster(self):
        extractive_summary = self.extractive_summarizer.summarize(self.data, self.query)
        self.abstractive_rewording = self.abstractive_worder.reword(extractive_summary)

    def generate_summary(self):
        formatted_summary = self.format(self.abstractive_rewording)
        self.save(formatted_summary)

    def format(self, summary):
        #Perform formatting if needed
        self.save(summary)

    def save(self, summary):
        with open(self.save_path, 'w') as f:
            for word in summary:
                f.write(word)


if __name__ == '__main__':
    args = sys.argv
    pipeline = Pipeline(args)
    pipeline.evaluate_cluster()
    pipeline.generate_summary()