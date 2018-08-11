from utils.decoder import CoNLLDataset
from .preprocess import pre_process_CoNLLDataset
from .paramaters import paths


class HMM:
    def __init__(self, number_of_history_chars=3, dataset='CoNLL2003'):
        self.number_of_history_chars = number_of_history_chars

        if dataset == 'CoNLL2003':
            self.train = CoNLLDataset(paths.CoNLLDataset_train_path)
            self.test = CoNLLDataset(paths.CoNLLDataset_train_path)

            self.train_chars, self.train_labels = pre_process_CoNLLDataset(self.train)
        elif dataset == 'Sport5':
            pass
        else:
            raise NotImplementedError
