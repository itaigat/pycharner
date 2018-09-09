from models.HMM import HMM
from models.MEMM import MEMM
from models.NaiveClassifier import NaiveClassifier
from utils.decoder import SportDataset
from models.paramaters import DatasetsPaths


if __name__ == "__main__":
    # for i in range(2, 7):
    #     tst = HMM(number_of_history_chars=i, dataset='CoNLL2003', model_name= 'NumOfHistoryTry' + str(i))
    # nv = NaiveClassifier()

    feature_name_list = [('0_Label_1', '3_Type_1', '3_Type_2', '1_Char_3', '1_Char_4', '1_Char_5', '1_Char_6',
                          '1_Char_7', '2_Pos_1', '2_Pos_4', '2_Pos_5', '2_Pos_6', '2_Pos_7'),
                         ('0_Label_1', '3_Type_1', '3_Type_2', '1_Char_3', '1_Char_4', '1_Char_5', '2_Pos_1', '2_Pos_4',
                          '2_Pos_5'),
                         ('0_Label_1', '1_Char_1', '1_Char_2', '1_Char_3'), ]
    tst = MEMM(number_of_history_chars=7,
               number_of_history_pos=7,
               number_of_history_types=4,
               number_of_history_labels=2,
               regularization_factor=2.0,
               feature_name_list=feature_name_list,
               dataset='CoNLL2003')

    # feature_name_list_hebrew = [('0_Label_1', '1_Char_1', '1_Char_2', '1_Char_3'),
    #                             ('0_Label_1', '2_Binyan_1', '2_Binyan_4', '2_Binyan_5'),
    #                             ('0_Label_1', '4_Gender_1', '4_Gender_2')
    #                             ]
    #
    # tst = MEMM(number_of_history_chars=7,
    #            number_of_history_pos=7,
    #            number_of_history_types=4,
    #            number_of_history_labels=2,
    #            regularization_factor=2.0,
    #            feature_name_list=feature_name_list_hebrew,
    #            dataset='Sport5',
    #            reverse=True)
