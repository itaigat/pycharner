from utils.decoder import CoNLLDataset
from utils import score
from .preprocess import pre_process_CoNLLDataset
from .preprocess import pre_process_CoNLLDataset_for_score_test
from .paramaters import paths
from models.algorithms import Viterbi

class HMM:
    def __init__(self, number_of_history_chars=5, dataset='CoNLL2003'):
        self.number_of_history_chars = number_of_history_chars

        if dataset == 'CoNLL2003':
            self.train = CoNLLDataset(paths.CoNLLDataset_train_path)
            self.test = CoNLLDataset(paths.CoNLLDataset_train_path)
            self.valid = CoNLLDataset(paths.CoNLLDataset_valid_path)

            self.train_chars, self.train_labels = pre_process_CoNLLDataset(self.train)
            self.test_chars, self.test_labels = pre_process_CoNLLDataset(self.test)
            self.valid_chars, self.valid_labels = pre_process_CoNLLDataset(self.valid, row_limit=None)

        elif dataset == 'Sport5':
            pass
        else:
            raise NotImplementedError


        state_prior_dict, transition_dict, emission_dict, smoothing_factor = self.create_all_probabilities_for_viterbi(self.train_chars,
                                                                                                     self.train_labels,
                                                                                                     number_of_history_chars)
        # devel_obs = self.create_obs_list(self.valid_chars, number_of_history_chars)
        # # print(transition_dict)
        # vt_res = Viterbi.viterbi_for_hmm(obs=tuple(devel_obs),
        #                                  states=tuple(set(self.train_labels)),
        #                                  start_p=state_prior_dict,
        #                                  trans_p=transition_dict,
        #                                  emit_p=emission_dict,
        #                                  non_history_obs='_' * number_of_history_chars,
        #                                  smoothing_factor=smoothing_factor)
        # print('Viterbi Ended With Proba: ' + str(vt_res[0]))
        # output_words, output_pred = score.turn_char_predictions_to_word_predictions(tuple(devel_obs),vt_res[1])

        output_words, output_pred = self.test_dataset(
                                             dataset_chars = self.valid_chars,
                                             number_of_history_chars = number_of_history_chars,
                                             states = set(self.train_labels),
                                             state_prior_dict = state_prior_dict,
                                             transition_dict = transition_dict,
                                             emission_dict = emission_dict,
                                             non_history_obs = '_' * number_of_history_chars,
                                             smoothing_factor = smoothing_factor)
        actual_words, actual_pred = pre_process_CoNLLDataset_for_score_test(self.valid, row_limit=None)
        if tuple(output_words) != tuple(actual_words):
            print ('Word align critical problem !!')
        report_str = 'Run Summary:\nNumber of history chars in test:' + str(number_of_history_chars) + '\n'

        for label_type in (list(set(actual_pred)) + ['ALL']):
            p_score = score.precision(output_pred, actual_pred, e_type=label_type)
            r_score = score.recall(output_pred, actual_pred, e_type=label_type)
            f1_score = score.F1(output_pred, actual_pred, e_type = label_type)
            report_str += "For " + str(label_type) + ":\nPrecision: " + str(p_score) + "\n"
            report_str += "Recall: " + str(r_score) + "\nF1 Score:" + str(f1_score) + "\n"
        print(str(report_str))
        with open('HMM_Run_Summary.txt', 'w') as f:
            f.write(report_str)





    def create_all_probabilities_for_viterbi(self,
                                             characters,
                                             char_labels,
                                             history_len,
                                             smoothed = True):

            """
            :param characters: list of characters (of train set)
            :param char_labels: list of characters labels (of train set)
            :param history_len: the number of characters used for history
            :return: For train set creates all state prior probabilities, transition probabilities
                     and emission_probability for viterbi
            """

            state_prior_dict = {}
            transition_dict = {}
            emission_dict = {}
            observations_state_counter_dict = {}

            total_chars = 0.0
            all_states = list(set(char_labels))
            for state in all_states:
                state_prior_dict[state] = 0.0

            history_list = ['_'] * history_len
            memory_label = None
            for i in range(len(characters)):
                curr_char = characters[i]
                curr_label = char_labels[i]

                if curr_char == '\n':
                    # means new document
                    history_list = ['_'] * history_len
                    memory_label = None
                    continue

                total_chars += 1
                # fills state_prior_dict in the number of times each label appeared
                state_prior_dict[curr_label] += 1
                # fills transition_dict for each label the number of times given this label appears each label after
                if memory_label is not None:
                    if memory_label in transition_dict:
                        if curr_label in transition_dict[memory_label]:
                            transition_dict[memory_label][curr_label] += 1.0
                        else:
                            transition_dict[memory_label][curr_label] = 1.0
                    else:
                        transition_dict[memory_label] = {}
                        transition_dict[memory_label][curr_label] = 1.0

                # in the next section smoothing needs to be added
                history_sequence = ''
                for char in history_list:
                    history_sequence += char
                curr_sequence = history_sequence[1:] + curr_char
                given_data = (history_sequence, curr_label)
                # fills observations_state_counter_dict with the number of times the history sequence appears with
                # the current label (given_data)
                if given_data in observations_state_counter_dict:
                    observations_state_counter_dict[given_data] += 1.0
                else:
                    observations_state_counter_dict[given_data] = 1.0
                # fills emission_dict with the number of times the given_data leads to curr_sequence
                if given_data in emission_dict:
                    if curr_sequence in emission_dict[given_data]:
                        emission_dict[given_data][curr_sequence] += 1.0
                    else:
                        emission_dict[given_data][curr_sequence] = 1.0
                else:
                    emission_dict[given_data] = {}
                    emission_dict[given_data][curr_sequence] = 1.0
                # at end of loop
                history_list.pop(0)
                history_list.append(curr_char)
                memory_label = curr_label

            # creates probabilities from all data
            # for transition probabilities
            for outer_state in transition_dict:
                for inner_state in transition_dict[outer_state]:
                    transition_dict[outer_state][inner_state] = transition_dict[outer_state][inner_state] / float(
                        state_prior_dict[outer_state])
                for state in all_states:
                    # adds 0 probability to unreachable states
                    if state not in transition_dict[outer_state]:
                        transition_dict[outer_state][state] = 0.0
            if smoothed == True:
                num_of_unique_chars = len(set(characters))

            # for prior probabilities
            for state in state_prior_dict:
                state_prior_dict[state] = state_prior_dict[state] / float(total_chars)
            # for emission probabilities: (smoothing needed)
            for given_data in emission_dict:
                num_of_existing_options = 0
                for predicted_data in emission_dict[given_data]:
                    num_of_existing_options += 1
                    emission_dict[given_data][predicted_data] = emission_dict[given_data][predicted_data] / float(
                        observations_state_counter_dict[given_data])
                if smoothed == True:
                    # factor to normalize probabilities
                    factor = 1 + (num_of_unique_chars - num_of_existing_options) / float(num_of_unique_chars)
                    for predicted_data in emission_dict[given_data]:
                            emission_dict[given_data][predicted_data] = emission_dict[given_data][predicted_data] / float( factor )
                    emission_dict[given_data]['Smoothing Factor'] = (1.0/num_of_unique_chars)/factor

            if smoothed == True:
                smooth_factor = 1.0/num_of_unique_chars
            else:
                smooth_factor = 0.0

            return state_prior_dict, transition_dict, emission_dict, smooth_factor



    def create_obs_list(self,
                        characters,
                        history_len):
        """
        :param characters: list of characters
        :param history_len: length of history used characters
        :return: observation list ready for viterbi
        """
        observations = []
        history_obs = '_' * history_len
        # observations.append(history_obs)
        for i in range(len(characters)):
            curr_char = characters[i]

            if curr_char == '\n':
                # means new document
                history_obs = '_' * history_len
                observations.append(history_obs)
                continue

            curr_obs = history_obs[1:] + curr_char
            observations.append(curr_obs)
            history_obs = curr_obs

        return observations

    def test_dataset(self,
                     dataset_chars,
                     number_of_history_chars,
                     states,
                     state_prior_dict,
                     transition_dict,
                     emission_dict,
                     non_history_obs,
                     smoothing_factor):
        """
        :param dataset_chars: list of chars
        :param states: set of states
        :param state_prior_dict: prior probabilities for each state
        :param transition_dict: transition probabilities between states
        :param emission_dict: emission probabilities dict
        :param non_history_obs: the begining of a sentence history observation
        :param smoothing_factor: smoothing factor for unseen emission forms
        :return: list of words with list of word predictions
        """
        output_words = []
        output_pred = []
        num_of_processed_sent = 0
        temp_chars = []
        for char in dataset_chars:
            if char == '\n':
                obs_for_viterbi = self.create_obs_list(temp_chars , number_of_history_chars)
                vt_res = Viterbi.viterbi_for_hmm(obs=tuple(obs_for_viterbi),
                                                 states=tuple(states),
                                                 start_p=state_prior_dict,
                                                 trans_p=transition_dict,
                                                 emit_p=emission_dict,
                                                 non_history_obs=non_history_obs,
                                                 smoothing_factor=smoothing_factor)
                temp_output_words, temp_output_pred = score.turn_char_predictions_to_word_predictions(obs_for_viterbi, vt_res[1])
                output_words.extend(temp_output_words)
                output_pred.extend(temp_output_pred)
                num_of_processed_sent += 1
                print('Sentence Processed :' + str(num_of_processed_sent) + ' , Viterbi Proba: ' + str(vt_res[0]))

                temp_chars = []
            else:
                temp_chars.append(char)

        return output_words, output_pred