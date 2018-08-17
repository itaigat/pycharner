from models.HMM import HMM
from models.MEMM import MEMM
from models.NaiveClassifier import NaiveClassifier


if __name__ == "__main__":
    # for i in range(2, 7):
    #     tst = HMM(number_of_history_chars=i, dataset='CoNLL2003', model_name= 'NumOfHistoryTry' + str(i))
    # nv = NaiveClassifier()
    # feature_name_list = ['0_Label', ('1_Char_1', '2_Pos_1', '3_Type_1')]
    feature_name_list = ['0_Label',('1_Char_1','3_Type_2'),('3_Type_1', '2_Pos_1')]
    tst = MEMM(number_of_history_chars = 2,
               number_of_history_pos = 2,
               number_of_history_types = 2,
               number_of_history_labels = 1,
               regularization_factor = 1.5,
               feature_name_list = feature_name_list,
               dataset = 'CoNLL2003')

