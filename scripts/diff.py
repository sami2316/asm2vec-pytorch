import os
import torch
import torch.nn as nn
import argparse
import bisect
import time
import asm2vec
import logger
import numpy as np
import _pickle as cPickle
from scipy.spatial import distance

parser = argparse.ArgumentParser()
parser.add_argument("-d1", "--database1_dir", help="Extracted database directory for first Binary")
parser.add_argument("-d2", "--database2_dir", help="Extracted database directory for second Binary")
parser.add_argument("-r", "--results_dir", help="Directory name to save match results")
parser.add_argument("-m", "--model", help="Trained/Saved model.pt")
args = parser.parse_args()

def cosine_similarity(v1, v2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(v1, v2).item()

class Asm2Vec(object):
    def __init__(self, ipath1, ipath2, mpath, epochs=10, device='auto', lr=0.02):
        self.logger = logger.get_logger(self.__class__.__name__)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info("Device : %s" % device)
        
        # Load model
        self.model, self.tokens = asm2vec.utils.load_model(mpath, device=device)
        # Load Data
        ipaths = [ipath1, ipath2]
        functions, tokens_new = asm2vec.utils.load_data(ipaths, full=False)
        self.tokens.update(tokens_new)
        self.model.update(len(functions), self.tokens.size())
        self.model = self.model.to(device)

        # train function embedding
        self.model = asm2vec.utils.train(
            functions,
            self.tokens,
            model=self.model,
            epochs=epochs,
            device=device,
            mode='test',
            learning_rate=lr
        )

    def get_model(self):
        return self.model

class Dataset(object):
    def __init__(self, ipath, model):
        self.logger = logger.get_logger(self.__class__.__name__)
        self.function_vectors = []
        self.function_names = []
        self.function_count = 0
        self.model = model
    
        self.function_names, self.functions, tokens_new = asm2vec.utils.load_data(ipath, full=True)
        self.generate()

    def __len__(self):
        return self.function_count

    def generate(self):
        self.function_count = len(self.functions)
        for i,fn in enumerate(self.functions):
            fn_tensor = self.model.to('cpu').embeddings_f(torch.tensor([i])).detach().numpy()
            self.function_vectors.append(fn_tensor)

    def remove(self, index):
        del self.function_names[index]
        del self.function_vectors[index]
        self.function_count -= 1

class Matcher(object):
    def __init__(self, dataset1_dir, dataset2_dir, results_dir, model_path):
        self.logger = logger.get_logger(self.__class__.__name__)
        self.dataset1_dir = dataset1_dir
        self.dataset2_dir = dataset2_dir
        self.results_dir = results_dir
        self.model_path = model_path
        self.matches = []
        handler = Asm2Vec(dataset1_dir, dataset2_dir, model_path)
        self.model = handler.get_model()

    def _handle_match(self, dataset1, dataset2, index1, index2, distance):
        match = ( 
            dataset1.function_names[index1],
            dataset2.function_names[index2],
            distance )

        if distance <= 0.5:
            bisect.insort_left(self.matches, match)
        else:
            return

        # Update the datasets
        dataset1.remove(index1)
        dataset2.remove(index2)

    # Filter the sub functions and strings (Special case for Radare2)
    def filter_func(self, in_dataset):
        for index in range(len(in_dataset) -1, -1, -1):
            func_name = in_dataset.function_names[index]
            if func_name.startswith('str.') or func_name.startswith('loc.'):
                in_dataset.remove(index)


    def _match_function_vectors(self, index1, dataset1, dataset2):
        # 1- Convert to numpy data of comparable dimensions
        target = dataset1.function_vectors[index1].reshape(1,-1)
        vectors = np.vstack(dataset2.function_vectors)

        try:
            distances = distance.cdist(target, vectors, 'cosine')[0]
        except:
            distances = None

        if np.all(np.isnan(distances)):
            return None
        return distances

    def _match(self, dataset1, dataset2):
        num_matches = 0
        total = len(dataset1.function_names)
        self.logger.info("Algorithm will start matching functions...")
        for index1 in range(len(dataset1) -1, -1, -1):
            distances = self._match_function_vectors(index1, dataset1, dataset2)
            if distances is None:
                continue

            min_index = np.nanargmin(distances)
            min_distance = distances[min_index]
            min_count = np.count_nonzero(distances == min_distance)
            if min_count == 1: 
                num_matches += 1
                self.logger.debug('Function Matched (%d / %d) : %s' % (num_matches, total, dataset1.function_names[index1]))
                self._handle_match(dataset1, dataset2, index1, min_index, min_distance)

        return num_matches

    # main matching function
    def match_all(self):
        self.logger.info("Loading Datasets for comparison .... ")
        dataset1 = Dataset(self.dataset1_dir, self.model)
        dataset2 = Dataset(self.dataset2_dir, self.model)
        self.logger.info("Datasets Loaded...")

        # Filter irrelevent functions
        #self.filter_func(dataset1)
        #self.filter_func(dataset2)

        start = int(time.time())
        num_matches = self._match(dataset1, dataset2)
        end = int(time.time())

        self.logger.debug('Found a total of %d matches in %d seconds' % (num_matches, end-start))

        #
        # Create directory where results will be written to.
        #
        if not os.access(self.results_dir, os.F_OK):
            os.makedirs(self.results_dir)

        #
        # Save the functions that were successfully matched.
        #
        self.logger.debug('Saving matched functions')

        with open('%s/matched.pcl' % self.results_dir, 'wb') as fp:
            for match in self.matches:
                cPickle.dump(match, fp)

        #
        # During the matching process, matched elements are removed from their
        # datasets. Consequently, when we get here, `dataset1' and `dataset2'
        # hold only those elements that weren't matched. Save them for later
        # reference. These are the "primary unmatched" and "secondary unmatched"
        # sets.
        #
        self.logger.debug('Saving primary unmatched functions')

        with open('%s/unmatched1.pcl' % self.results_dir, 'wb') as fp:
            for i in range(len(dataset1)):
                cPickle.dump((dataset1.function_names[i]), fp)

        self.logger.debug('Saving secondary unmatched functions')

        with open('%s/unmatched2.pcl' % self.results_dir, 'wb') as fp:
            for i in range(len(dataset2)):
                cPickle.dump((dataset2.function_names[i]), fp)
    
if __name__ == '__main__':
    if all(vars(args).values()):
        m = Matcher(args.database1_dir, args.database2_dir, args.results_dir, args.model) 
        m.match_all()
    else:
        parser.print_help()
