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
                 feature_name_list,
                 dataset='CoNLL2003'):
        self.number_of_history_chars = number_of_history_chars
        self.number_of_history_pos = number_of_history_pos
        self.number_of_history_types = number_of_history_types
        self.number_of_history_labels = number_of_history_labels

        self.number_of_gradient_decent_steps = 3
        self.learning_rate = 0.0003
        self.regularization_factor = regularization_factor
        self.feature_name_list = feature_name_list
        # '0_Label'
        if dataset == 'CoNLL2003':
            self.train = CoNLLDataset(paths.CoNLLDataset_train_path)
            self.test = CoNLLDataset(paths.CoNLLDataset_test_path)
            self.valid = CoNLLDataset(paths.CoNLLDataset_valid_path)

            self.train_chars, self.train_labels, self.train_pos = pre_process_CoNLLDataset(self.train, memm=True)
            self.valid_chars, self.valid_labels, self.valid_pos = pre_process_CoNLLDataset(self.valid, row_limit=None,
                                                                                           memm=True)

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
            dataset_chars=self.valid_chars,
            dataset_pos=self.valid_pos,
            dataset_types=self.valid_types,
            history_len_char=self.number_of_history_chars,
            history_len_pos=self.number_of_history_pos,
            history_len_type=self.number_of_history_types,
            history_len_labels=self.number_of_history_labels,
            states=set(self.train_labels),
            probabilities_dict=self.train_probabilities,
            non_history_state=('O', 'S'),
            smoothing_factor_dict=self.smoothing_factor_dict,
            feature_name_list=self.feature_name_list)

        model_name = 'MEMM_{0}_{1}_{2}_{3}_{4}'.format(str(dataset), str(self.number_of_history_chars), str(
            self.number_of_history_pos), str(self.number_of_history_types), str(self.number_of_history_labels))

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
        :param feature_name_list: 
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

        state_obs_occurrences = {}
        obs_occurrences = {}
        all_features = []
        all_states = list(set(char_labels))
        for state in all_states:
            state_obs_occurrences[state] = {}

        history_char_list = ['_'] * history_len_char
        history_pos_list = ['_'] * history_len_pos
        history_type_list = ['_'] * history_len_type
        history_label_list = [('O', 'S')] * history_len_label

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
                history_label_list = [('O', 'S')] * history_len_label
                continue

            history_char_list.append(curr_char)
            history_pos_list.append(curr_pos)
            history_type_list.append(curr_type)

            feature_obs = (
                tuple(history_label_list), tuple(history_char_list), tuple(history_pos_list), tuple(history_type_list))

            # counts for each state how many times each feature observation led to it
            for feature_kind in feature_name_list:
                curr_feature = create_feature_from_observation(feature_kind, feature_obs)
                # obs_index = int(feature_kind[0])
                if curr_feature not in state_obs_occurrences[curr_label]:
                    state_obs_occurrences[curr_label][curr_feature] = 1
                else:
                    state_obs_occurrences[curr_label][curr_feature] += 1
                # counts for each feature observation how many times it appeared
                if curr_feature not in obs_occurrences:
                    obs_occurrences[curr_feature] = 1
                else:
                    obs_occurrences[curr_feature] += 1

                all_features_count_dict[feature_kind] += 1
                if 'Char' not in feature_kind:
                    non_char_feat = True
                    for feat in feature_kind:
                        if 'Char' in feat:
                            non_char_feat = False
                    if non_char_feat:
                        # chars wont get negative weights
                        all_features.append(curr_feature)

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
        for state in state_obs_occurrences:
            for feature in all_features:
                if feature not in state_obs_occurrences[state]:
                    state_obs_occurrences[state][feature] = 0.0

        state_feature_weight_dict = self.calc_feature_weights(all_states,
                                                              obs_occurrences,
                                                              state_obs_occurrences)

        print('Finished weight calc...')
        # first = True
        # create probabilities from for all p(state | obs)
        state_obs_probability_dict = {}
        for outer_state in all_states:
            # if first == True:
            #     print(state_feature_weight_dict[outer_state])
            state_obs_probability_dict[outer_state] = {}
            for obs in state_feature_weight_dict[outer_state]:
                probability_denominator = 0.0
                for inner_state in all_states:
                    if obs in state_feature_weight_dict[inner_state]:
                        probability_denominator += exp(state_feature_weight_dict[inner_state][obs])
                # creates the probability - might need smoothing
                state_obs_probability_dict[outer_state][obs] = exp(state_feature_weight_dict[outer_state][obs]) / float(
                    probability_denominator)
            # if first == True:
            #     print('\n')
            #     print(state_obs_probability_dict[outer_state])
            #     first = False
        print('Finished probability calc...')
        smoothing_factor_dict = {}
        for feature_kind in feature_name_list:
            non_char_feat = True
            for feat in feature_kind:
                if 'Char' in feat:
                    non_char_feat = False
            if non_char_feat:
                smoothing_factor_dict[feature_kind] = 1 / float(len(set(characters)))
            else:
                smoothing_factor_dict[feature_kind] = 1 / float(all_features_count_dict[feature_kind])

        return state_obs_probability_dict, smoothing_factor_dict

    def calc_feature_weights(self,
                             all_states,
                             feature_count_dict,
                             state_feature_count_dict):
        """
        :param all_states: list of all possible states
        :param feature_count_dict: dictionary containing how many times each feature observation occurred in train
        :param state_feature_count_dict: dictionary containing how many times each state has each feature observation
         lead to it in train
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
                            # print('Not Supposed To Get Here')
                            denominator += feature_count_dict[outer_obs] * exp(0)

                    for inner_state in all_states:
                        if outer_obs in state_feature_count_dict[inner_state]:
                            numerator = state_feature_count_dict[inner_state][outer_obs] * exp(curr_weight)
                            expected_count += numerator / float(denominator)
                    # the last part of the formula is to avoid overfitting
                    curr_partial_derivative = float(curr_empirical_count) - expected_count - curr_weight / float(
                        self.regularization_factor)
                    # if curr_weight != 0.0:
                    #     print (str(curr_weight ))
                    state_feature_weight_dict[state][
                        outer_obs] = curr_weight + self.learning_rate * curr_partial_derivative
            print('Gradient Accent Step Finished: [' + str(step + 1) + " : " + str(
                self.number_of_gradient_decent_steps) + "]")
        return state_feature_weight_dict

    def create_obs_list(self,
                        characters,
                        char_pos,
                        char_types,
                        history_len_char,
                        history_len_pos,
                        history_len_type,
                        feature_name_list):
        """
        :param feature_name_list:
        :param characters: list of chars
        :param char_pos: list of cher's word's part of speech
        :param char_types: list of cher's types
        :param history_len_char: how many chars in history
        :param history_len_pos: how many part of speech in history
        :param history_len_type: how many char's types in history
        :return: observation list ready for viterbi
        """
        observations = []
        obs_for_score = []
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
            # label list is empty currently
            feature_obs = [[], tuple(history_char_list), tuple(history_pos_list), tuple(history_type_list)]
            obs_for_score.append(feature_obs)

            feature_obs_processed = []
            for feature_kind in feature_name_list:
                curr_feature = create_feature_from_observation(feature_kind, feature_obs, no_labels=True)
                feature_obs_processed.append(curr_feature)

            observations.append(feature_obs_processed)

            history_char_list.pop(0)
            history_pos_list.pop(0)
            history_type_list.pop(0)

        return observations, obs_for_score

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
        :param feature_name_list:
        :param smoothing_factor_dict:
        :param non_history_state:
        :param probabilities_dict:
        :param history_len_labels:
        :param history_len_type:
        :param history_len_pos:
        :param history_len_char:
        :param dataset_types:
        :param dataset_pos:
        :param dataset_chars: list of chars
        :param states: set of states
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
                obs_for_viterbi, obs_for_score = self.create_obs_list(
                    characters=temp_chars,
                    char_pos=temp_pos,
                    char_types=temp_types,
                    history_len_char=history_len_char,
                    history_len_pos=history_len_pos,
                    history_len_type=history_len_type,
                    feature_name_list=feature_name_list)

                vt_res = Viterbi.viterbi_for_memm(obs=tuple(obs_for_viterbi),
                                                  states=tuple(states),
                                                  train_probabilities=probabilities_dict,
                                                  non_history_label=non_history_state,
                                                  number_of_history_labels=history_len_labels,
                                                  smoothing_factor_dict=smoothing_factor_dict,
                                                  feature_name_list=feature_name_list,
                                                  create_feature_from_observation=create_feature_from_observation)

                temp_output_words, temp_output_pred = score.turn_char_predictions_to_word_predictions(obs_for_score,
                                                                                                      vt_res[1],
                                                                                                      memm=True)
                output_words.extend(temp_output_words)
                output_pred.extend(temp_output_pred)
                num_of_processed_sent += 1
                print(
                    'Sentence Processed :' + str(num_of_processed_sent) + ' , Viterbi Probabilities: ' + str(vt_res[0]))

                temp_chars = []
                temp_pos = []
                temp_types = []
            else:
                temp_chars.append(dataset_chars[i])
                temp_pos.append(dataset_pos[i])
                temp_types.append(dataset_types[i])

        return output_words, output_pred


def create_feature_from_observation(feature_detail, observation, no_labels=False):
    """
     :param no_labels:
     :param feature_detail: fature detail from feature name list
     :param observation: current observation
     :return: create required feature from current observation
     """
    if not isinstance(feature_detail, tuple):
        obs_index = int(feature_detail[0])
        feature = observation[obs_index]
    else:
        feature = []
        for inner_feat in feature_detail:
            if no_labels and ('Label' in inner_feat):
                feature.append(None)
                continue
            obs_index = int(inner_feat[0])
            inner_index = (-1) * int(inner_feat.split('_')[2])
            feature.append(observation[obs_index][inner_index])
        if not no_labels:
            feature = tuple(feature)
    return feature
