import math
import numpy as np

class Extractive:
    def __init__(self):
        pass
    
    def summarize(self, data, query):
        return "summarized"

class TfIdf:
    def __init__(self, doc_path, query=None, extraction_quota=0.5):
        self.doc_path = doc_path
        if extraction_quota>1.0:
            self.extraction_quota=0.5 # TODO: Find neater way to do this
            print("Bad extraction quota. Must be <= 1.0. Using 0.5 instead")
        else:
            self.extraction_quota=extraction_quota
        
        with open(self.doc_path, 'r') as f:
            self.paragraphs = f.readlines() # TODO: Make split on paragraphs instead of newlines.
        self.nd = len(self.paragraphs) # Number of paragraphs
        self.l = int(math.floor(len(self.paragraphs)*self.extraction_quota)) # Number of paragraphs to extract
        self.query = query

    def perform_extraction(self, save_path):
        self.calc_ranks()
        with open(save_path, 'w') as f:
            for paragraph in self.ranked_paragraphs[0:self.l]: # Select the l highest ranked paragraphs
                f.write(paragraph+'\n')
        
        print("Extraction complete: ", len(self.paragraphs)-self.l, " paragraphs removed.")

    def calc_ranks(self):
        scores = []
        for paragraph in self.paragraphs:
            score = self.calc_score(paragraph)
            scores.append(score)
        rank_set = zip(scores,self.paragraphs)
        sorted_rank_set = sorted(rank_set, reverse=True)

        _, self.ranked_paragraphs = zip(*sorted_rank_set) # Paragraphs ordered according to their rank

    def calc_score(self, paragraph):
        score = 0
        for word in self.query:
            score += self.calc_subscore(word,paragraph)

        return score # Score for each paragraph

    def calc_subscore(self, word, paragraph):
        ndw = 0
        for paragraph in self.paragraphs:
            if paragraph.contains(word):
                ndw+=1
        
        nw = 0
        for w in paragraph:
            if w==word:
                nw+=1

        return nw * np.log(self.nd/ndw) # Score of a word in query string for a given paragraph