from models.HMM import HMM
from models.NaiveClassifier import NaiveClassifier


if __name__ == "__main__":
    for i in range(2, 7):
        tst = HMM(number_of_history_chars=5, dataset='CoNLL2003', model_name= 'NumOfHistoryTry' + str(i))
    # nv = NaiveClassifier()
