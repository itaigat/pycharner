types = ['MISC', 'ORG', 'PER', 'LOC']

def precision(predicted, true, e_type='ALL'):
    """
    Returns the amount of relevant relevant retrieved documents divided by the amount of retrieved documents
    :param predicted: List of the predicted results
    :param true: List of the true labels
    :param e_type: The entity type we want to check
    :return: precision score
    """

    true_counter = 0
    retrieved_counter = 0

    if len(predicted) != len(true) or len(true) == 0 or len(predicted) == 0:
        return 0.0

    if e_type == 'ALL':
        for i, value in enumerate(predicted):
            if predicted[i] == true[i] != 'O':
                true_counter += 1
            if predicted[i] != 'O':
                retrieved_counter += 1
        # retrieved_counter = len(predicted)
    elif e_type == 'BINARY':
        for i, value in enumerate(predicted):
            if predicted[i] in types and true[i] in types:
                true_counter += 1
            if predicted[i] in types:
                retrieved_counter += 1
    else:
        for i, value in enumerate(predicted):
            if predicted[i] == e_type:
                retrieved_counter += 1
                if predicted[i] == true[i]:
                    true_counter += 1

    return float(true_counter) / retrieved_counter


def recall(predicted, true, e_type='ALL'):
    """
    Returns the amount of relevant relevant retrieved documents divided by the amount of relevant documents
    :param predicted: List of the predicted results
    :param true: List of the true labels
    :param e_type: The entity type we want to check
    :return: recall score
    """
    true_counter = 0
    relevant_counter = 0

    if len(predicted) != len(true) or len(true) == 0 or len(predicted) == 0:
        return 0.0

    if e_type == 'ALL':
        for i in range(len(predicted)):
            if predicted[i] == true[i] != 'O':
                true_counter += 1
            if true[i] != 'O':
                relevant_counter += 1
        # relevant_counter = len(true)
    elif e_type == 'BINARY':
        for i, value in enumerate(predicted):
            if predicted[i] in types and true[i] in types:
                true_counter += 1
            if true[i] in types:
                relevant_counter += 1
    else:
        for i in range(len(predicted)):
            if predicted[i] == true[i] == e_type:
                true_counter += 1
            if true[i] == e_type:
                relevant_counter += 1

    return float(true_counter) / relevant_counter


def F1(predicted, true, e_type='ALL'):
    """
    F1 calculated by the formula 2 * ((Precision * Recall) / (Precision + Recall)
    :param predicted: List of the predicted results
    :param true: List of the true labels
    :param e_type: The entity type we want to check
    :return: recall score
    """
    precision_score = precision(predicted, true, e_type)
    recall_score = recall(predicted, true, e_type)

    return 2 * ((precision_score * recall_score) / (precision_score + recall_score))


def turn_char_predictions_to_word_predictions(observations, char_predictions, memm=False, reverse=False):
    """
    :param reverse:
    :param memm:
    :param observations: obs list as entered to the hmm viterbi
    :param char_predictions: list of prediction that is the viterbi output
    :return: list of words and list of word predictions
    """
    words = []
    predictions = []

    curr_word = ''
    curr_pred = ''
    for i in range(len(observations)):
        if memm:
            curr_char = observations[i][1][len(observations[i][1]) - 1]
        else:
            curr_char = observations[i][-1:]
        pred = char_predictions[i]

        if curr_char == '_':
            # when there is a space move to other word
            if curr_word != '':
                if reverse:
                    curr_word = curr_word[::-1]
                words.append(curr_word)
                predictions.append(curr_pred)
            curr_word = ''
            curr_pred = ''
            continue

        if curr_pred == '':
            curr_pred = pred[0]

        curr_word += curr_char

        if curr_pred != pred[0]:
            print('Unexpected prediction problem got: {0} in same word of {1}'.format(str(pred), curr_pred))

    if curr_word != '':
        if reverse:
            curr_word = curr_word[::-1]
        words.append(curr_word)
        predictions.append(curr_pred)

    return words, predictions


def check_all_results_parameters(model_name,
                                 output_words,
                                 actual_words,
                                 output_pred,
                                 actual_pred,
                                 number_of_history_chars):
    """
    :param model_name: the model name (HMM and etc)
    :param output_words: the model output list of words
    :param actual_words: the actual list of words
    :param output_pred: the model prediction per word
    :param actual_pred: the actual labeling
    :param number_of_history_chars: number of history characters
    :return: creates report and and puts in the repository
    """
    prediction_types = list(set(actual_pred))
    prediction_types.append('ALL')
    prediction_types.append('BINARY')

    if tuple(output_words) != tuple(actual_words):
        print("Output len:" + str(len(output_words)) + " Actual len:" + str(len(actual_words)))
        print('Word align critical problem !!')

    report_str = 'Run Summary:\n Number of history chars in test:' + str(number_of_history_chars) + '\n'

    for label_type in prediction_types:
        p_score = precision(output_pred, actual_pred, e_type=label_type)
        r_score = recall(output_pred, actual_pred, e_type=label_type)
        f1_score = F1(output_pred, actual_pred, e_type=label_type)
        report_str += "For " + str(label_type) + ": \n Precision: " + str(p_score) + "\n"
        report_str += "Recall: " + str(r_score) + "\n F1 Score:" + str(f1_score) + "\n"
    print(str(report_str))

    with open(str(model_name) + '_Run_Summary.txt', 'w') as f:
        f.write(report_str)
