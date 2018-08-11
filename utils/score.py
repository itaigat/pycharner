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
        for i in range(len(predicted)):
            if predicted[i] == true[i]:
                true_counter += 1
        retrieved_counter = len(predicted)
    else:
        for i in range(len(predicted)):
            if predicted[i] == e_type:
                retrieved_counter += 1
                if predicted[i] == true[i]:
                    true_counter += 1

    return true_counter / retrieved_counter


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
            if predicted[i] == true[i]:
                true_counter += 1
        relevant_counter = len(true)
    else:
        for i in range(len(predicted)):
            if predicted[i] == true[i] == e_type:
                true_counter += 1
            if true[i] == e_type:
                relevant_counter += 1

    return true_counter / relevant_counter


def F1(predicted, true, e_type='ALL'):
    """
    F1 calculated by the formula 2 * ((Precision * Recall) / (Precision + Recall)
    :param predicted: List of the predicted results
    :param true: List of the true labels
    :param e_type: The entity type we want to check
    :return: recall score
    """
    precision_score = precision(predicted, true, e_type)
    recall_score = precision(predicted, true, e_type)

    return 2 * ((precision_score * recall_score) / (precision_score + recall_score))

def turn_char_predictions_to_word_predictions(observations,char_predictions):
    """
    :param observations: obs list as entered to the hmm viterbi
    :param char_predictions: list of prediction that is the viterbi output
    :return: list of words and list of word predictions
    """
    words = []
    predictions = []

    curr_word = ''
    curr_pred = ''
    for i in range(len(observations)):
        curr_char = observations[i][-1:]
        pred = char_predictions[i]

        if curr_char == '_':
            # when there is a space move to other word
            if curr_word != '':
                words.append(curr_word)
                predictions.append(curr_pred)
            curr_word = ''
            curr_pred = ''
            continue

        if curr_pred == '':
            curr_pred = pred[0]

        curr_word += curr_char

        if curr_pred != pred[0]:
            print('Unexpected predicrion problem got:' + str(pred) + ' in same word of ' + curr_pred)

    return words, predictions





