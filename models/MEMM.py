from utils.decoder import CoNLLDataset
from utils import score
from .preprocess import pre_process_CoNLLDataset
from .preprocess import pre_process_CoNLLDataset_for_score_test
from .preprocess import create_string_type_tagging
from .paramaters import paths
from models.algorithms import Viterbi
from math import exp

class MEMM:
    def __init__(self,
                 number_of_history_chars,
                 number_of_history_pos,
                 number_of_history_types,
                 number_of_history_labels,
                 regularization_factor,
                 dataset='CoNLL2003'):
        
        self.number_of_history_chars = number_of_history_chars
        self.number_of_history_pos = number_of_history_pos
        self.number_of_history_types = number_of_history_types
        self.number_of_history_labels = number_of_history_labels

        self.number_of_gradient_decent_steps = 15
        self.learning_rate = 0.0001
        self.regularization_factor = regularization_factor
        self.feature_name_list = ['0_Label', '1_Char', '2_Pos', '3_Type']

        if dataset == 'CoNLL2003':
            self.train = CoNLLDataset(paths.CoNLLDataset_train_path)
            self.test = CoNLLDataset(paths.CoNLLDataset_test_path)
            self.valid = CoNLLDataset(paths.CoNLLDataset_valid_path)

            self.train_chars, self.train_labels, self.train_pos = pre_process_CoNLLDataset(self.train)
            # self.test_chars, self.test_labels, self.test_pos = pre_process_CoNLLDataset(self.test)
            self.valid_chars, self.valid_labels, self.valid_pos = pre_process_CoNLLDataset(self.valid, row_limit=None)

        elif dataset == 'Sport5':
            pass
        else:
            raise NotImplementedError

        self.train_types = create_string_type_tagging(self.train_chars)

        self.train_probabilities, self.smoothing_factor_dict = self.create_all_probabilities_for_viterbi(
            characters=self.train_chars,
            char_labels=self.train_labels,
            char_pos=self.train_pos,
            char_types=self.train_types,
            history_len_char=self.number_of_history_chars,
            history_len_pos=self.number_of_history_pos,
            history_len_type=self.number_of_history_types,
            history_len_label=self.number_of_history_labels,
            feature_name_list=self.feature_name_list
            )

        self.valid_types = create_string_type_tagging(self.valid_chars)

        output_words, output_pred = self.test_dataset(
            dataset_chars = self.valid_chars,
            dataset_pos = self.valid_pos,
            dataset_types = self.valid_types,
            history_len_char = self.number_of_history_chars,
            history_len_pos = self.number_of_history_pos,
            history_len_type = self.number_of_history_types,
            history_len_labels = self.number_of_history_labels,
            states = set(self.train_labels),
            probabilities_dict = self.train_probabilities,
            non_history_state = ('O' , 'F'),
            smoothing_factor_dict = self.smoothing_factor_dict,
            feature_name_list = self.feature_name_list)

        model_name = 'MEMM_' + str(dataset) + '_' + str(self.number_of_history_chars) + '_' + str(self.number_of_history_pos) + '_' \
                     + str(self.number_of_history_types) + '_' + str(self.number_of_history_labels)

        actual_words, actual_pred = pre_process_CoNLLDataset_for_score_test(self.valid, row_limit=None)

        score.check_all_results_parameters(model_name=model_name,
                                           output_words=output_words,
                                           actual_words=actual_words,
                                           output_pred=output_pred,
                                           actual_pred=actual_pred,
                                           number_of_history_chars=number_of_history_chars)





    def create_all_probabilities_for_viterbi(self,
                                             characters,
                                             char_labels,
                                             char_pos,
                                             char_types,
                                             history_len_char,
                                             history_len_pos,
                                             history_len_type,
                                             history_len_label,
                                             feature_name_list):
        """
        :param characters: list of chars
        :param char_labels: list of cher's labels
        :param char_pos: list of cher's word's part of speech
        :param char_types: list of cher's types
        :param history_len_char: how many chars in history
        :param history_len_pos: how many part of speech in history
        :param history_len_type: how many char's types in history
        :param history_len_label: how many char's label in history
        :return: for each state and feature observation in train dict containing p( state | feature observation )
        """

        state_obs_accurences = {}
        obs_accurences = {}
        all_features = []
        all_states = list(set(char_labels))
        for state in all_states:
            state_obs_accurences[state] = {}

        history_char_list = ['_'] * history_len_char
        history_pos_list = ['_'] * history_len_pos
        history_type_list = ['_'] * history_len_type
        history_label_list = [('O', 'F')] * history_len_label


        all_features_count_dict = {}
        for feature_kind in feature_name_list:
            all_features_count_dict[feature_kind] = 0.0

        for i in range(len(characters)):
            curr_char = characters[i]
            curr_label = char_labels[i]
            curr_type = char_types[i]
            curr_pos = char_pos[i]

            if curr_char == '\n':
                # means new document
                history_char_list = ['_'] * history_len_char
                history_pos_list = ['_'] * history_len_pos
                history_type_list = ['_'] * history_len_type
                history_label_list = [('O', 'F')] * history_len_label
                continue

            history_char_list.append(curr_char)
            history_pos_list.append(curr_pos)
            history_type_list.append(curr_type)


            feature_obs = (tuple(history_label_list) , tuple(history_char_list), tuple(history_pos_list) , tuple(history_type_list))

            # counts for each state how many times each feature observation led to it
            for feature_kind in feature_name_list:
                obs_index = int(feature_kind[0])
                if feature_obs[obs_index] not in state_obs_accurences[curr_label]:
                    state_obs_accurences[curr_label][feature_obs[obs_index]] = 1
                else:
                    state_obs_accurences[curr_label][feature_obs[obs_index]] += 1
                # counts for each feature observation how many times it appeared
                if feature_obs[obs_index] not in obs_accurences:
                    obs_accurences[feature_obs[obs_index]] = 1
                else:
                    obs_accurences[feature_obs[obs_index]] += 1

                all_features_count_dict[feature_kind] += 1
                all_features.append(feature_obs[obs_index])

            # each train observation is a tuple containing the state and the feature vector that leads to it
            # all_observations.append( (curr_label, feature_obs) )

            # at end of loop
            if history_len_label > 0:
                history_label_list.pop(0)
                history_label_list.append(curr_label)

            history_char_list.pop(0)
            history_pos_list.pop(0)
            history_type_list.pop(0)
        # adds to every state zero counts for features that didn't appear with it
        all_features = list(set(all_features))
        for state in state_obs_accurences:
            for feature in all_features:
                if feature not in state_obs_accurences[state]:
                    state_obs_accurences[state][feature] = 0.0

        state_feature_weight_dict = self.calc_feature_weights(all_states,
                                                              obs_accurences,
                                                              state_obs_accurences)

        print ('Finished wieght calc...')
        first = True
        # create probabilities from for all p(state | obs)
        state_obs_probability_dict = {}
        for outer_state in all_states:
            # if first == True:
            #     print(state_feature_weight_dict[outer_state])
            state_obs_probability_dict[outer_state] = {}
            for obs in state_feature_weight_dict[outer_state]:
                proba_denominator = 0.0
                for inner_state in all_states:
                    if obs in state_feature_weight_dict[inner_state]:
                        proba_denominator += exp(state_feature_weight_dict[inner_state][obs])
                # creates the probability - might nood smoothing
                state_obs_probability_dict[outer_state][obs] = exp(state_feature_weight_dict[outer_state][obs]) / float(
                    proba_denominator)
            # if first == True:
            #     print('\n')
            #     print(state_obs_probability_dict[outer_state])
            #     first = False
        print('Finished probability calc...')
        smoothing_factor_dict = {}
        for feature_kind in feature_name_list:
            smoothing_factor_dict = 1/float(all_features_count_dict[feature_kind])
        return state_obs_probability_dict, smoothing_factor_dict

    def calc_feature_weights(self,
                             all_states,
                             feature_count_dict,
                             state_feature_count_dict):
        """
        :param all_states: list of all possible states
        :param feature_count_dict: dictionary containing how many times each feature observation accured in train
        :param state_feature_count_dict: dictionary containing how many times each state has each feature observation lead to it in train
        :return: for each state and observation in state_feature_count_dict dict containing learned weight
        """
        state_feature_weight_dict = {}
        # init all weights to 0
        for state in state_feature_count_dict:
            state_feature_weight_dict[state] = {}
            for obs in state_feature_count_dict[state]:
                state_feature_weight_dict[state][obs] = 0.0

        # the loop below preforms gradient accent to find the state-feature weights
        for step in range(self.number_of_gradient_decent_steps):
            for state in state_feature_weight_dict:
                for outer_obs in state_feature_weight_dict[state]:
                    # the last step (state, obs) weight
                    curr_weight = state_feature_weight_dict[state][outer_obs]
                    # the number of times obs lead to state
                    curr_empirical_count = state_feature_count_dict[state][outer_obs]
                    # expected_count from the partial derivative formula
                    expected_count = 0.0
                    denominator = 0.0
                    for inner_state in all_states:
                        if outer_obs in state_feature_count_dict[inner_state]:
                            amount_appeared_together = state_feature_count_dict[inner_state][outer_obs]
                            denominator += amount_appeared_together * exp(curr_weight)
                            denominator += (feature_count_dict[outer_obs] - amount_appeared_together) * exp(0)
                        else:
                            print('Not Supposed To Get Here')
                            denominator += feature_count_dict[outer_obs] * exp(0)

                    for inner_state in all_states:
                        if outer_obs in state_feature_count_dict[inner_state]:
                            numerator = state_feature_count_dict[inner_state][outer_obs] * exp(curr_weight)
                            expected_count += numerator/float(denominator)
                    # the last part of the formula is to avoid overfitting
                    curr_partial_derivative = float(curr_empirical_count) - expected_count - curr_weight/float(self.regularization_factor)

                    state_feature_weight_dict[state][outer_obs] = curr_weight + self.learning_rate * curr_partial_derivative
            print('Gradient Accent Step Finished: ' + str(step + 1))
        return state_feature_weight_dict

    def create_obs_list(self,
                        characters,
                        char_pos,
                        char_types,
                        history_len_char,
                        history_len_pos,
                        history_len_type,):
        """
        :param characters: list of chars
        :param char_pos: list of cher's word's part of speech
        :param char_types: list of cher's types
        :param history_len_char: how many chars in history
        :param history_len_pos: how many part of speech in history
        :param history_len_type: how many char's types in history
        :return: observation list ready for viterbi
        """
        observations = []
        history_char_list = ['_'] * history_len_char
        history_pos_list = ['_'] * history_len_pos
        history_type_list = ['_'] * history_len_type

        for i in range(len(characters)):
            curr_char = characters[i]
            curr_type = char_types[i]
            curr_pos = char_pos[i]

            if curr_char == '\n':
                # means new document
                history_char_list = ['_'] * history_len_char
                history_pos_list = ['_'] * history_len_pos
                history_type_list = ['_'] * history_len_type
                continue

            history_char_list.append(curr_char)
            history_pos_list.append(curr_pos)
            history_type_list.append(curr_type)

            feature_obs = [[], tuple(history_char_list), tuple(history_pos_list), tuple(history_type_list)]
            observations.append(feature_obs)

            history_char_list.pop(0)
            history_pos_list.pop(0)
            history_type_list.pop(0)

        return observations

    def test_dataset(self,
                     dataset_chars,
                     dataset_pos,
                     dataset_types,
                     history_len_char,
                     history_len_pos,
                     history_len_type,
                     history_len_labels,
                     states,
                     probabilities_dict,
                     non_history_state,
                     smoothing_factor_dict,
                     feature_name_list):
        """
        :param number_of_history_chars:
        :param dataset_chars: list of chars
        :param states: set of states
        :param state_prior_dict: prior probabilities for each state
        :param transition_dict: transition probabilities between states
        :param emission_dict: emission probabilities dict
        :param non_history_obs: the beginning of a sentence history observation
        :param smoothing_factor: smoothing factor for unseen emission forms
        :return: list of words with list of word predictions
        """
        output_words = []
        output_pred = []
        num_of_processed_sent = 0
        temp_chars = []
        temp_pos = []
        temp_types = []
        for i in range(len(dataset_chars)):
            if dataset_chars[i] == '\n':
                obs_for_viterbi = self.create_obs_list(
                                                 characters=temp_chars,
                                                 char_pos=temp_pos,
                                                 char_types=temp_types,
                                                 history_len_char=history_len_char,
                                                 history_len_pos=history_len_pos,
                                                 history_len_type=history_len_type)

                vt_res = Viterbi.viterbi_for_memm(obs=tuple(obs_for_viterbi),
                                                 states=tuple(states),
                                                 train_probabilities=probabilities_dict,
                                                 non_history_label=non_history_state,
                                                 number_of_history_labels=history_len_labels,
                                                 smoothing_factor_dict=smoothing_factor_dict,
                                                 feature_name_list=feature_name_list)

                temp_output_words, temp_output_pred = score.turn_char_predictions_to_word_predictions(obs_for_viterbi,
                                                                                                      vt_res[1],
                                                                                                      memm = True)
                output_words.extend(temp_output_words)
                output_pred.extend(temp_output_pred)
                num_of_processed_sent += 1
                print('Sentence Processed :' + str(num_of_processed_sent) + ' , Viterbi Proba: ' + str(vt_res[0]))

                temp_chars = []
                temp_pos = []
                temp_types = []
            else:
                temp_chars.append(dataset_chars[i])
                temp_pos.append(dataset_pos[i])
                temp_types.append(dataset_types[i])

        return output_words, output_pred