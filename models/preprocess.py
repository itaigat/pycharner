def pre_process_CoNLLDataset(dataset, row_limit = None):
    """
    :param dataset: A Dataset object
    :return: returns two lists:
    characters - list of text characters in order where spaces are replaced with '_'
    char_labels - list of labels for each character (for characters[i] the label is char_labels[i])
    labels correspond to the article description
    """
    rows = dataset.__str__().split('\n')
    words = []
    word_labels = []

    if row_limit is not None:
        num_rows = row_limit
    else:
        num_rows = len(rows) - 1

    for i in range(0,num_rows, 2):
        # TODO: add '/n' handling
        words.extend(rows[i].split(' ') + ['\n'])
        word_labels.extend(rows[i + 1].split(' ') + ['O'])

    if len(words) != len(word_labels):
        print('pre_process_CoNLLDataset problem - words and word labels are not alienged')
        raise ValueError

    characters = []
    char_labels = []

    for j in range(len(words)):
        word = words[j]
        word_label = word_labels[j]

        if 'LOC' in word_label:
            word_label = 'LOC'
        elif 'MISC' in word_label:
            word_label = 'MISC'
        elif 'ORG' in word_label:
            word_label = 'ORG'
        elif 'PER' in word_label:
            word_label = 'PER'
        for i in range(len(word)):
            characters.append(word[i])
            if i == (len(word) - 1):
                # The final char of the word gets label (ord_label, 'F')
                char_labels.append((word_label, 'F'))
            else:
                char_labels.append((word_label, i + 1))

        # Add spaces
        characters.append('_')
        # check if this is the correct state
        char_labels.append(('O', 'F'))

    return characters, char_labels

def pre_process_CoNLLDataset_for_score_test(dataset, row_limit = None):
    """
    :param dataset: Dataset object
    :param row_limit: number of rows to process
    :return: word list and list of true word labels
    """
    rows = dataset.__str__().split('\n')
    words = []
    word_labels = []

    if row_limit is not None:
        num_rows = row_limit
    else:
        num_rows = len(rows) - 1
    for i in range(0, num_rows, 2):
        # TODO: add '/n' handling -no need
        words.extend(rows[i].split(' ') )
        word_labels.extend(rows[i + 1].split(' ') )

    for j in range(len(word_labels)):
        word_label = word_labels[j]
        if 'LOC' in word_label:
            word_label = 'LOC'
        elif 'MISC' in word_label:
            word_label = 'MISC'
        elif 'ORG' in word_label:
            word_label = 'ORG'
        elif 'PER' in word_label:
            word_label = 'PER'
        word_labels[j] = word_label

    return words, word_labels