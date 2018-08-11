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
