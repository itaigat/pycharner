from utils.decoder import CoNLLDataset, SportDataset
from utils.common import zfill_files
from models.algorithms import Viterbi


if __name__ == "__main__":
    """
    ds = CoNLLDataset('data\\CoNLL2003\\test.txt')
    print(ds)
    ds = SportDataset('data\\Sport5\\data')
    print(ds)

    zfill_files('data\\Sport5\\data')
    """

    vt = Viterbi.example()
