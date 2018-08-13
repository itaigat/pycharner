from models.HMM import HMM
from models.NaiveClassifier import NaiveClassifier


if __name__ == "__main__":
    tst = HMM()
    print(NaiveClassifier().train_model())

