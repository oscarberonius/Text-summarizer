from models.abstractive import Abstractive
from models.extractive import Extractive, TfIdf
from utils.data_processing import DataProcessor

from os.path import dirname
import json
import sys
import tarfile
import numpy as np

class Pipeline:

    def __init__(self, vscodearg=None, loaded_data_file_path=None, compressed_data_path=None):
        self.base_path = dirname(__file__)
        self.data_processor = DataProcessor()
        self.loaded_data_file_path = loaded_data_file_path
        self.compressed_data_path = compressed_data_path

    def init_data(self):
        if not self.loaded_data_file_path and self.compressed_data_path : # No data file provided, process from data set
            fp = self.base_path+self.compressed_data_path
            print(f'Processing compressed data set from {fp}')
            self.data = self.data_processor.process_data(fp)
            print(f"Data loaded from {fp}")
            print(f"Dps: {len(self.data)}")
        elif self.loaded_data_file_path:
            fp = self.base_path+self.loaded_data_file_path
            print(f'Loading processed data from {fp}')
            if tarfile.is_tarfile(fp): # Load data if its tarred
                tar = tarfile.open(fp)
                info = tar.getmembers()
                f = tar.extractfile(info[0])
                self.data = json.load(f)
                f.close()
            else:                     # Load normally otherwise
                with open(fp, 'rb') as f:
                    self.data = json.load(f)
            print(f'Data loaded from {fp}')
            print(f"Dps: {len(self.data)}")
        else: 
            print('No data path specified.')

    def generate_data_file(self, filename):
        fp = self.base_path+'/data/'+filename
        print(f'Start file generation')
        with open(fp, 'w') as f:
            json.dump(self.data, f)
            print(f'Data file generated at {fp}')
        
    def clean_data(self, text_size_threshold=2000, ingress_size_threshold=300, low_end=0.01, high_end=0.3):
        size_before_cleaning=len(self.data)
        print('~~Cleaning data~~')
        print('Remove dps containing unwanted tokens')
        data = self.data_processor.clean_data(self.data)
        print('Remove unbalanced texts')
        data = self.data_processor.remove_unbalanced_texts(data, low_end, high_end)
        print('Remove large texts')
        self.data = self.data_processor.remove_large_texts(data, text_size_threshold, ingress_size_threshold)
        print('~~Cleaning complete~~')
        print(f'Number of dps {size_before_cleaning} -> {len(self.data)}')

    def create_input_and_target_files(self, nbr_of_files):
        nbr_of_files = len(self.data) - 1 if nbr_of_files  >= len(self.data) else nbr_of_files
        sample_indices = np.random.rand(nbr_of_files)
        sample_indices = np.floor(sample_indices*len(self.data)).astype(int)

        with open(self.base_path+"/data/input_large.txt", 'w') as f:
            for index in sample_indices:
                line = self.data[index]['text']
                line = line.replace('\n', '') + '\n'
                f.write(line.encode('utf-8'))

        with open(self.base_path+"/data/target_large.txt", 'w') as f:
            for index in sample_indices:
                line = self.data[index]['ingress']
                line = line.replace('\n', '') + '\n'
                f.write(line.encode('utf-8'))
        print("Input and target files created")

    def test_extractive(self, query, extraction_quota):
        path = self.base_path+'/data/input_large.txt'
        query = "council australian social media win world Woolworths"
        idf = TfIdf(path, query, extraction_quota=0.5)
        idf.perform_extraction(self.base_path+'/data/extractive_test_output.txt')

    def visualize_data(self):
        self.data_processor.visualize_data_point_sizes(self.data)

if __name__ == '__main__':
    args = sys.argv
    pipeline = Pipeline(compressed_data_path='/data/en-ingress-dataset', loaded_data_file_path='/data/test.json')
    pipeline.init_data()
    pipeline.clean_data()
    pipeline.generate_data_file('test.json')
    #pipeline.visualize_data()
    #pipeline.create_input_and_target_files(300000000)