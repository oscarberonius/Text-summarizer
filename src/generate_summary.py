from models.abstractive import Abstractive
from models.extractive import Extractive, TfIdf
from utils.data_processing import DataProcessor

from os.path import dirname
import json
import sys
import tarfile
import numpy as np

class Pipeline:

    def __init__(self, vscodearg=None, data_path='/data/data_set_latest', query=None, save_path='/data/summary.txt'): 
        self.path = dirname(__file__)
        self.data_processor = DataProcessor()
    #    self.data = self.data_processor.process_data(self.path+data_path)
        #print("saving file")
  #     with open(self.path+'/data/latest_data.json', 'wr') as f:
   #         json.dump(self.data, f)


        self.query = query
        self.save_path = save_path
        text_size_threshold = 3000
        ingress_size_threshold = 300
        low_end = 0.01
        high_end = 0.3

        #tar = tarfile.open(self.path+'/data/clean_data1.tar.gz')
        #info = tar.getmembers()
        #file = tar.extractfile(info1[0])
        with open(self.path+ '/data/latest_data.json', 'r') as j:
            obj = json.load(j)
        data = obj

        # tar1 = tarfile.open(self.path+'/data/clean_data1.tar.gz')
        # info1 = tar1.getmembers()
        # file1 = tar1.extractfile(info1[0])
        # obj1 = json.load(file1)

        # tar2 = tarfile.open(self.path + '/data/clean_data2.tar.gz')
        # info2 = tar2.getmembers()
        # file2 = tar2.extractfile(info2[0])
        # obj2 = json.load(file2)

        #data = obj1+obj2
        data = self.data_processor.clean_data(data)
        data = self.data_processor.remove_unbalanced_texts(data, low_end, high_end)
        self.data = self.data_processor.remove_large_texts(data, text_size_threshold, ingress_size_threshold)
          #   self.abstractive_worder = Abstractive(self.data)

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

    def create_input_and_target_files(self, nbr_of_files):
        nbr_of_files = len(self.data) - 1 if nbr_of_files  >= len(self.data) else nbr_of_files
        sample_indices = np.random.rand(nbr_of_files)
        sample_indices = np.floor(sample_indices*len(self.data)).astype(int)

        with open(self.path+"/data/input.txt", 'w') as f:
            for index in sample_indices:
                line = self.data[index]['text']
                line = line.replace('\n', '') + '\n'
                f.write(line.encode('utf-8'))

        with open(self.path+"/data/target.txt", 'w') as f:
            for index in sample_indices:
                line = self.data[index]['ingress']
                line = line.replace('\n', '') + '\n'
                f.write(line.encode('utf-8'))
        print("Input and target files created")

    def test_extractive(self, query, extraction_quota):
        path = self.path+'/data/input.txt'
        query = "council australian social media win world Woolworths"
        idf = TfIdf(path, query, extraction_quota=0.5)
        idf.perform_extraction(self.path+'/data/extractive_test_output.txt')
    
    def visualize_data(self):
        self.data_processor.visualize_data_point_sizes(self.data)

if __name__ == '__main__':
    args = sys.argv
    pipeline = Pipeline(args)

 #   pipeline.visualize_data()
    pipeline.create_input_and_target_files(300000000)
#    pipeline.evaluate_cluster()
#    pipeline.generate_summary()