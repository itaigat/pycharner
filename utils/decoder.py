import math
from os import listdir, path
from xml.etree import ElementTree


class Dataset(object):
    """Class that iterates over Dataset
    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags
    If processing_word and processing_tag are not None,
    optional preprocessing is applied
    Example:
        ```python
        data = Dataset(filename)
        for sentence, tags in data:
            pass
        ```
    """

    def __init__(self, filename, max_iter=None):
        """
        Args:
            filename: path to the file
            max_iter: (optional) max number of sentence to yield
        """
        self.filename = filename
        self.max_iter = max_iter
        self.length = None

    def __iter__(self):
        pass

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

    def __str__(self):
        st = ''
        for sentence, tag, pos in self:
            st += ' '.join(sentence) + '\n' + ' '.join(tag) + '\n' + ' '.join(pos) + '\n'

        return st


class CoNLLDataset(Dataset):
    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags, poss = [], [], []
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags, poss
                        words, tags, poss = [], [], []
                else:
                    ls = line.split(' ')
                    word, tag, pos = ls[0], ls[-1], ls[1]

                    words += [word]
                    tags += [tag]
                    poss += [pos]

                    """
                    In case you want a word and not a sentence - take off the comment below
                    yield [word], [tag] 
                    """


class SportDataset(Dataset):
    def __init__(self, filename, features=None, part='train', train_size=0.8, valid_size=0.1, test_size=0.1):
        super().__init__(filename)

        self.features = features
        self.part = part

        if train_size + test_size + valid_size != 1:
            raise ValueError

        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size

    def __iter__(self):
        files = [f for f in listdir(self.filename)]

        train_len = math.floor(len(files) * self.train_size)
        test_len = math.floor(len(files) * self.test_size) + train_len

        if self.part == 'train':
            files = files[:train_len]
        elif self.part == 'test':
            files = files[train_len:test_len]
        elif self.part == 'valid':
            files = files[test_len:]
        elif self.part == 'all':
            pass
        else:
            raise ValueError

        for file in files:
            with open(path.join(self.filename, file), encoding='utf8') as f:
                xml = ElementTree.parse(f).getroot()[0]  # Prior - There is no file that contains more than one article

                words, tags = [], []
                features = self.init_features(self.features)
                for paragraph in xml:
                    for sentence in paragraph:
                        for word in sentence:
                            words += [word.attrib['surface']]

                            try:
                                tags += [word[0][0][0].attrib['type']]
                            except KeyError:
                                tags += ['None']
                            except IndexError:
                                tags += ['None']

                            for feature in self.features:
                                try:
                                    features[feature] += [word[0][0][0].attrib[feature]]
                                except KeyError:
                                    features[feature] += ['None']
                                except IndexError:
                                    features[feature] += ['None']

                    yield words, tags, features

                    words, tags = [], []
                    features = self.init_features(self.features)

    @staticmethod
    def init_features(features):
        dic = {}
        for feature in features:
            dic[feature] = []

        return dic
