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
        for sentence, tag in self:
            st += ' '.join(sentence) + '\n' + ' '.join(tag) + '\n'

        return st


class CoNLLDataset(Dataset):
    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0], ls[-1]

                    words += [word]
                    tags += [tag]

                    """
                    In case you want a word and not a sentence - take off the comment below
                    yield [word], [tag] 
                    """


class SportDataset(Dataset):
    def __iter__(self):
        files = [f for f in listdir(self.filename)]

        for file in files:
            with open(path.join(self.filename, file), encoding='utf8') as f:
                xml = ElementTree.parse(f).getroot()[0]  # Prior - There is no file that contains more than one article

                words, tags = [], []
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

                    yield words, tags
                    words, tags = [], []