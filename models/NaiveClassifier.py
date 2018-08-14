import operator

from models.paramaters import paths
from utils.decoder import CoNLLDataset
from models.preprocess import pre_process_CoNLLDataset
from utils import score


class NaiveClassifier:
    def __init__(self, number_of_history_chars=5, dataset='CoNLL2003'):
        self.number_of_history_chars = number_of_history_chars

        if dataset == 'CoNLL2003':
            self.train = CoNLLDataset(paths.CoNLLDataset_train_path)
            self.valid = CoNLLDataset(paths.CoNLLDataset_valid_path)

            self.train_chars, self.train_labels, self.train_pos = pre_process_CoNLLDataset(self.train)
            self.valid_labels, self.valid_labels, self.valid_pos = pre_process_CoNLLDataset(self.valid)

        self.label = ''
        self.train_model()
        self.test_model()

    def train_model(self):
        counter = {'LOC': 0, 'PER': 0, 'ORG': 0, 'MISC': 0}
        for label in self.train_labels:
            if label[0] in counter.keys():
                counter[label[0]] += 1

        self.label = max(counter.items(), key=operator.itemgetter(1))[0]

    def test_model(self):
        score.check_all_results_parameters('Naive', (), (), [item[0] for item in self.valid_labels],
                                           [self.label] * len(self.valid_labels), 5)
