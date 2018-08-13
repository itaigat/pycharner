import operator

from models.paramaters import paths
from utils.decoder import CoNLLDataset
from models.preprocess import pre_process_CoNLLDataset


class NaiveClassifier:
    def __init__(self, number_of_history_chars=5, dataset='CoNLL2003'):
        self.number_of_history_chars = number_of_history_chars

        if dataset == 'CoNLL2003':
            self.train = CoNLLDataset(paths.CoNLLDataset_train_path)
            self.train_chars, self.train_labels = pre_process_CoNLLDataset(self.train)

    def train_model(self):
        counter = {'LOC': 0, 'PER': 0, 'ORG': 0, 'MISC': 0}
        for label in self.train_labels:
            if label[0] in counter.keys():
                counter[label[0]] += 1

        return int(max(counter.items(), key=operator.itemgetter(0))[1]) / sum(counter.values())
