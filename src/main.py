from models.abstractive import Abstractive
from models.extractive import Extractive, TfIdf
from utils.data_processing import DataProcessor

from os.path import dirname
import json
import argparse
import tarfile
import numpy as np
import os

class Pipeline:

    def __init__(self, vscodearg=None, loaded_data_file_path=None, compressed_data_path=None):
        self.base_path = dirname(__file__)
        self.data_processor = DataProcessor()
        self.loaded_data_file_path = loaded_data_file_path
        self.compressed_data_path = compressed_data_path
        self.data = None

    def init_data_files(self, fps):
        combined_data = []
        for fp in fps:
            with open(fp, 'r') as f:
                data = json.load(f)
                combined_data += data
        self.data = combined_data
        print(f'Data initialized with {len(self.data)} number of dps')

    def init_data(self):
        if not self.loaded_data_file_path and self.compressed_data_path:  # No data file provided, process from data set
            fp = os.path.join(self.base_path, self.compressed_data_path)
            print(f'Processing compressed data set from {fp}')
            self.data = self.data_processor.process_data(fp)
            print(f"Data loaded from {fp}")
            print(f"Dps: {len(self.data)}")
        elif self.loaded_data_file_path:
            fp = os.path.join(self.base_path, self.loaded_data_file_path)
            print(f'Loading processed data from {fp}')
            if tarfile.is_tarfile(fp):  # Load data if its tarred
                tar = tarfile.open(fp)
                info = tar.getmembers()
                f = tar.extractfile(info[0])
                self.data = json.load(f)
                f.close()
            else:  # Load normally otherwise
                with open(fp, 'r') as f:
                    self.data = json.load(f)
            print(f'Data loaded from {fp}')
            print(f"Dps: {len(self.data)}")
        else:
            print('No data path specified.')

    def generate_data_file(self, filename):
        fp = os.path.join(self.base_path, 'data/'+filename)
        print(f'Start file generation at {fp}')
        with open(fp, 'w') as f:
            json.dump(self.data, f)
            print(f'Data file generated at {fp}')

    def clean_data(self):
        size_before_cleaning = len(self.data)
        print('~~Cleaning data~~')
        print('Remove dps containing unwanted tokens')
        self.data = self.data_processor.clean_data(self.data)
        print('~~Cleaning complete~~')
        print(f'Number of dps {size_before_cleaning} -> {len(self.data)}')

    # Removes data points where number of tokens fall outside specified interval, as well as data points where the ratio
    # of number of tokens text/ingress is unbalanced given an interval
    def trim_tokens(self, text_low_end=1, text_high_end=2000, ingress_low_end=1, ingress_high_end=300, ratio_low_end=0.01, ratio_high_end=0.3):
        size_before_cleaning = len(self.data)
        print('~~Trimming data on number of tokens~~')
        print('Remove unbalanced dps')
        self.data = self.data_processor.remove_unbalanced_tokens(self.data, ratio_low_end, ratio_high_end)
        print(f'{size_before_cleaning - len(self.data)} dps removed.')
        self.data = self.data_processor.trim_tokens(self.data, text_low_end, text_high_end, ingress_low_end, ingress_high_end)
        print(f'{size_before_cleaning - len(self.data)} dps removed.')
        print(f'Number of dps {size_before_cleaning} -> {len(self.data)}')

    # Same as trim_tokens except measured by number of words instead of number of tokens
    def trim_words(self, text_low_end=1, text_high_end=350, ingress_low_end=1, ingress_high_end=50, ratio_low_end=0.01, ratio_high_end=0.3):
        size_before_cleaning = len(self.data)
        print('~~Trimming data on number of words~~')
        print('Remove unbalanced dps:')
        self.data = self.data_processor.remove_unbalanced_words(self.data, ratio_low_end, ratio_high_end)
        print(f'{size_before_cleaning - len(self.data)} dps removed.')
        print(f'Trim words:')
        self.data = self.data_processor.trim_words(self.data, text_low_end, text_high_end, ingress_low_end, ingress_high_end)
        print(f'{size_before_cleaning - len(self.data)} dps removed.')
        print(f'Number of dps {size_before_cleaning} -> {len(self.data)}')

    def create_input_and_target_files(self, nbr_of_files):
        nbr_of_files = len(self.data) - 1 if nbr_of_files >= len(self.data) else nbr_of_files
        sample_indices = np.random.rand(nbr_of_files)
        sample_indices = np.floor(sample_indices * len(self.data)).astype(int)

        with open(os.path.join(self.base_path,"data/input_.txt"), 'wb') as f:
            for index in sample_indices:
                line = self.data[index]['text']
                line = line.replace('\n', '') + '\n'
                f.write(line.encode('utf-8'))

        with open(os.path.join(self.base_path,"data/target_.txt"), 'wb') as f:
            for index in sample_indices:
                line = self.data[index]['ingress']
                line = line.replace('\n', '') + '\n'
                f.write(line.encode('utf-8'))
        print("Input and target files created")

    def test_extractive(self, query, extraction_quota):
        path = self.base_path + '/data/input_large.txt'
        query = "council australian social media win world Woolworths"
        idf = TfIdf(path, query, extraction_quota=0.5)
        idf.perform_extraction(self.base_path + '/data/extractive_test_output.txt')

    def visualize_token_count(self):
        self.data_processor.visualize_token_count(self.data)

    def visualize_word_count(self):
        self.data_processor.visualize_word_count(self.data)

    def remove_duplicates(self):
        self.data_processor.remove_duplicates(self.data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-loaded_data_file_path', required=False, default='final_data_nondup_clean_1.json')
    parser.add_argument('-compressed_data_path', required=False, default=None)
    opt = parser.parse_args()

    pipeline = Pipeline(loaded_data_file_path=opt.loaded_data_file_path, compressed_data_path=opt.compressed_data_path)
    #pipeline.init_data()

    pipeline.init_data_files(['final_data_nondup_clean_1.json', 'final_data_nondup_clean_2.json', 'final_data_nondup_clean_3.json', 'final_data_nondup_clean_4.json'])
    pipeline.trim_words()
    pipeline.trim_tokens()
    pipeline.visualize_token_count()
    #pipeline.clean_data_2() # clean_data - symbols, clean_data_2 - words
    #pipeline.generate_data_file('mars_data_1.json')
    #pipeline.clean_data()
    #pipeline.generate_data_file('mars_data_clean2.json')
    #pipeline.create_input_and_target_files(300000000)
    #pipeline.count_duplicates()


if __name__ == '__main__':
    main()
