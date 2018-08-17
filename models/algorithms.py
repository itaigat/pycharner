
class Viterbi:
    @staticmethod
    def print(V):
        for y in V[0].keys():
            print('State:', y)
            for t in range(len(V)):
                print(V[t][y])

    @staticmethod
    def viterbi(obs, states, start_p, trans_p, emit_p):
        V = [{}]
        path = {}
        for y in states:
            V[0][y] = start_p[y] * emit_p[y][obs[0]]
            path[y] = [y]
        for t in range(1, len(obs)):
            V.append({})
            newpath = {}
            for y in states:
                prob, state = max([(V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states])
                V[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath

        # Viterbi.print(V)
        prob, state = max([(V[len(obs) - 1][y], y) for y in states])

        return prob, path[state]

    @staticmethod
    def example():
        states = ('Rainy', 'Sunny')
        observations = ('walk', 'shop', 'clean')
        start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
        transition_probability = {
            'Rainy': {'Rainy': 0.7, 'Sunny': 0.3},
            'Sunny': {'Rainy': 0.4, 'Sunny': 0.6},
        }
        emission_probability = {
            'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
            'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
        }

        return Viterbi.viterbi(observations, states, start_probability, transition_probability, emission_probability)

    @staticmethod
    def viterbi_for_hmm(obs, states, start_p, trans_p, emit_p, non_history_obs, smoothing_factor):
        V = [{}]
        path = {}
        for y in states:
            if ((non_history_obs, y) in emit_p):
                if (obs[0] in emit_p[(non_history_obs, y)]):
                    V[0][y] = start_p[y] * emit_p[(non_history_obs, y)][obs[0]]
                else:
                    V[0][y] = start_p[y] * emit_p[(non_history_obs, y)]['Smoothing Factor']
            else:
                V[0][y] = start_p[y] * smoothing_factor
            path[y] = [y]
        print('Viterbi Number Of Obs:' + str(len(obs)))
        for t in range(1, len(obs)):
            if t % 1000 == 0:
                print('Viterbi Number Of Obs Processed:' + str(t))
            V.append({})
            newpath = {}
            for y in states:
                max_prob = - 1
                former_state = None
                for y0 in states:
                    if ((obs[t - 1], y) in emit_p):
                        if (obs[t] in emit_p[(obs[t - 1], y)]):
                            cur_prob = V[t - 1][y0] * trans_p[y0][y] * emit_p[(obs[t - 1], y)][obs[t]]
                        else:
                            cur_prob = V[t - 1][y0] * trans_p[y0][y] * emit_p[(obs[t - 1], y)]['Smoothing Factor']
                    else:
                        cur_prob = V[t - 1][y0] * trans_p[y0][y] * smoothing_factor

                    if cur_prob > max_prob:
                        max_prob = cur_prob
                        former_state = y0
                V[t][y] = max_prob
                newpath[y] = path[former_state] + [y]

            path = newpath

        prob = -1
        for y in states:
            cur_prob = V[len(obs) - 1][y]
            if cur_prob > prob:
                prob = cur_prob
                state = y

        # prob, state = max([(V[len(obs) - 1][y], y) for y in states])

        return prob, path[state]

    @staticmethod
    def viterbi_for_memm(obs, states, train_probabilities, non_history_label ,number_of_history_labels, smoothing_factor_dict, feature_name_list,create_feature_from_observation):
        V = [{}]
        path = {}
        curr_obs = obs[0]
        curr_obs[0] = tuple( [non_history_label] * number_of_history_labels )
        curr_obs = tuple(curr_obs)
        for y in states:
            curr_prob = 1.0
            for feature_kind in feature_name_list:
                curr_feature = create_feature_from_observation(feature_kind, curr_obs)
                # feature_indx = int(feature_kind[0])
                if curr_feature in train_probabilities[y]:
                    # print ('State: ' + str(y) + ' Feat:' + str(curr_obs[feature_indx]) + " Proba:"  + str(train_probabilities[y][curr_obs[feature_indx]]))
                    curr_prob = curr_prob * train_probabilities[y][curr_feature]
                else:
                    curr_prob = curr_prob * smoothing_factor_dict[feature_kind]
            # for probability not run to zero
            curr_prob = curr_prob * (10)
            V[0][y] = curr_prob
            # if ( curr_obs in train_probabilities[y] ):
            #     V[0][y] = train_probabilities[y][curr_obs]
            # elif y[1] == 1 or y[1] == 'F':
            #     V[0][y] = smoothing_factor
            # else:
            #     V[0][y] = 0.0
            path[y] = [y]
        print('Viterbi Number Of Obs:' + str(len(obs)))
        for t in range(1, len(obs)):
            if t % 1000 == 0:
                print('Viterbi Number Of Obs Processed:' + str(t))
            V.append({})
            newpath = {}
            curr_obs = obs[t]
            for y in states:
                max_prob = - 1
                former_state = None
                for y0 in states:
                    history_states = path[y0][t - number_of_history_labels :t][:]
                    if len(history_states) < number_of_history_labels:
                        history_states = [non_history_label]*(number_of_history_labels - len(history_states)) + history_states
                    temp_curr_obs = curr_obs[:]
                    temp_curr_obs[0] = tuple(history_states)
                    temp_curr_obs = tuple(temp_curr_obs)
                    curr_prob = V[t - 1][y0]
                    for feature_kind in feature_name_list:
                        curr_feature = create_feature_from_observation(feature_kind, temp_curr_obs)
                        # feature_indx = int(feature_kind[0])
                        if curr_feature in train_probabilities[y]:
                            curr_prob = curr_prob * train_probabilities[y][curr_feature]
                        else:
                            curr_prob = curr_prob * smoothing_factor_dict[feature_kind]
                    # for probability not run to zero
                    curr_prob = curr_prob * (10)
                    # if temp_curr_obs in train_probabilities[y]:
                    #     cur_prob = V[t - 1][y0] * train_probabilities[y][temp_curr_obs]
                    # elif((y0[1] == 'F') and (y == ('O','F'))) or ((y0[1] != 'F') and (y[0] == y0[0]) and (y[1] == 'F' or y[1] == y0[1] - 1)) :
                    #     cur_prob = V[t - 1][y0] * smoothing_factor
                    # elif (y == ('O','F')):
                    #     cur_prob = V[t - 1][y0] * smoothing_factor/2
                    # else:
                    #     cur_prob = V[t - 1][y0] * 0.0

                    if curr_prob > max_prob:
                        max_prob = curr_prob
                        former_state = y0
                V[t][y] = max_prob
                newpath[y] = path[former_state] + [y]

            path = newpath

        prob = -1
        for y in states:
            cur_prob = V[len(obs) - 1][y]
            if cur_prob > prob:
                prob = cur_prob
                state = y

        # prob, state = max([(V[len(obs) - 1][y], y) for y in states])

        return prob, path[state]