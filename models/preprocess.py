def pre_process_CoNLLDataset(dataset):
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
    for i in range(0, len(rows) - 1, 2):
        # need to add handling \n
        words.extend(rows[i].split(' '))
        word_labels.extend(rows[i + 1].split(' '))
    if len(words) != len(word_labels):
        print('pre_process_data problem - words and word labels are not alienged')
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
                # the final char of the word gets label (ord_label, 'F')
                char_labels.append((word_label, 'F'))
            else:
                char_labels.append((word_label, i + 1))
        # also add spaces
        characters.append('_')
        # check if this is the correct state
        char_labels.append(('O', 'F'))
    print(characters)
    print(char_labels)

    return characters, char_labels
