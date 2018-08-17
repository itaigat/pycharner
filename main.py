from models.HMM import HMM
from models.MEMM import MEMM
from models.NaiveClassifier import NaiveClassifier


if __name__ == "__main__":
    # for i in range(2, 7):
    #     tst = HMM(number_of_history_chars=i, dataset='CoNLL2003', model_name= 'NumOfHistoryTry' + str(i))
    # nv = NaiveClassifier()
    tst = MEMM(number_of_history_chars = 2,
               number_of_history_pos = 2,
               number_of_history_types = 2,
               number_of_history_labels = 2,
               regularization_factor = 2.0,
               dataset = 'CoNLL2003')

