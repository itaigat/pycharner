def pre_process_CoNLLDataset(dataset, row_limit=None, memm=False):
    """
    :param memm:
    :param dataset: A Dataset object
    :param row_limit: number of rows to process
    :return: returns two lists:
    characters - list of text characters in order where spaces are replaced with '_'
    char_labels - list of labels for each character (for characters[i] the label is char_labels[i])
    labels correspond to the article description
    """
    rows = dataset.__str__().split('\n')
    words = []
    word_labels = []
    word_pos = []

    if row_limit is not None:
        num_rows = row_limit
    else:
        num_rows = len(rows) - 1

    for i in range(0, num_rows, 3):
        words.extend(rows[i].split(' ') + ['\n'])
        word_labels.extend(rows[i + 1].split(' ') + ['O'])
        word_pos.extend(rows[i + 2].split(' ') + ['\n'])

    if len(words) != len(word_labels):
        print('pre_process_CoNLLDataset problem - words and word labels are not aligned')
        raise ValueError

    characters = []
    char_labels = []
    char_poss = []

    for j in range(len(words)):
        word = words[j]
        word_label = word_labels[j]
        char_pos = word_pos[j]

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
            char_poss.append(char_pos)

        if word == '\n':
            continue

        # Add spaces
        characters.append('_')
        # check if this is the correct state
        if memm:
            char_labels.append(('O', 'S'))
        else:
            char_labels.append(('O', 'F'))
        char_poss.append('_')

    return characters, char_labels, char_poss


def pre_process_CoNLLDataset_for_score_test(dataset, row_limit=None):
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
    for i in range(0, num_rows, 3):
        words.extend(rows[i].split(' '))
        word_labels.extend(rows[i + 1].split(' '))

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


def create_string_type_tagging(char_list):
    """
    :param char_list: list of characters
    :return: a list with the characters types
    """
    char_type_list = []
    for char in char_list:
        if char.isupper():
            char_type_list.append('X')
        elif char.islower():
            char_type_list.append('x')
        elif char.isdigit():
            char_type_list.append('d')
        else:
            char_type_list.append(char)

    return char_type_list

def create_string_type_tagging_word_base(char_list):
    """
    :param char_list: list of characters
    :return: a list with the characters types that are word based
    """
    char_type_list = []
    temp_word_len = 0
    temp_word_type = ''
    for char in char_list:

        if char == '_' or char == '\n':
            char_type_list.extend(([temp_word_type] * temp_word_len) + ['_'] )
            temp_word_len = 0
            temp_word_type = ''
            continue

        if char.isupper():
            if temp_word_type[-1:] != 'X':
                temp_word_type += 'X'
        elif char.islower():
            if temp_word_type[-1:] != 'x':
                temp_word_type += 'x'
        elif char.isdigit():
            if temp_word_type[-1:] != 'd':
                temp_word_type += 'd'
        else:
            if temp_word_type[-1:] != char:
                temp_word_type += char

        temp_word_len += 1

    if len(char_type_list) != len(char_list):
        print ("len(char_type_list) != len(char_list)")

    return char_type_list

def pre_process_Sport5Dataset(dataset, doc_limit=None):
    words_list = []
    word_labels = []
    features_dict = {}

    docs_processed = 0
    for words, tags, features in dataset:
        words_list.extend(words + ['\n'])
        word_labels.extend(tags + ['O'])
        for feature in features:
            if feature not in features_dict:
                features_dict[feature] = features[feature] + ['\n']
            else:
                features_dict[feature].extend(features[feature] + ['\n'])
        docs_processed += 1
        if doc_limit is not None and docs_processed >= doc_limit:
            break

    if len(words_list) != len(word_labels):
        print('pre_process_CoNLLDataset problem - words and word labels are not aligned')
        raise ValueError

    characters = []
    char_labels = []
    char_features = {}

    for feature in features_dict:
        char_features[feature] = []

    for j in range(len(words_list)):
        word = words_list[j]
        word_label = word_labels[j]

        if 'location' == word_label or 'country' == word_label or 'town' == word_label:
            word_label = 'LOC'
        elif 'symbol' == word_label:
            word_label = 'MISC'
        elif 'organization' == word_label:
            word_label = 'ORG'
        elif 'person' == word_label:
            word_label = 'PER'
        else:
            word_label = 'O'

        for i in range(len(word)):
            characters.append(word[i])
            if i == (len(word) - 1):
                # The final char of the word gets label (ord_label, 'F')
                char_labels.append((word_label, 'F'))
            else:
                char_labels.append((word_label, i + 1))

            for feature in features_dict:
                char_features[feature].append(features_dict[feature][j])

        if word == '\n':
            continue

        # Add spaces
        characters.append('_')
        char_labels.append(('O', 'S'))

        for feature in features_dict:
            char_features[feature].append('_')

    return characters, char_labels, char_features


def pre_process_Sport5Dataset_for_score_test(dataset, doc_limit=None):
    words_list = []
    word_labels = []

    docs_processed = 0
    for words, tags, features in dataset:
        words_list.extend(words)
        word_labels.extend(tags)

        docs_processed += 1
        if doc_limit is not None and docs_processed >= doc_limit:
            break

    if len(words_list) != len(word_labels):
        print('pre_process_CoNLLDataset problem - words and word labels are not aligned')
        raise ValueError
    # list of indexes to remove due to empty strings
    remove_index_list = []
    fix_index_list = []
    for j in range(len(word_labels)):
        if words_list[j] == '':
            remove_index_list.append(j)
        if ('_' in words_list[j]) and (len(words_list[j]) > 1):
            fix_index_list.append(j)
        word_label = word_labels[j]
        if 'location' == word_label or 'country' == word_label or 'town' == word_label:
            word_label = 'LOC'
        elif 'symbol' == word_label:
            word_label = 'MISC'
        elif 'organization' == word_label:
            word_label = 'ORG'
        elif 'person' == word_label:
            word_label = 'PER'
        else:
            word_label = 'O'
        word_labels[j] = word_label

    filtered_words_list = []
    filtered_word_labels = []
    for i in range(len(words_list)):
        if i in remove_index_list:
            continue
        elif i in fix_index_list:
            # for specific case that '_' in middle of the word
            broken_word = words_list[i].split('_')
            filtered_words_list.append(broken_word[0])
            filtered_words_list.append(broken_word[1])
            filtered_word_labels.append(word_labels[i])
            filtered_word_labels.append(word_labels[i])
            continue
        filtered_words_list.append(words_list[i])
        filtered_word_labels.append(word_labels[i])

    return filtered_words_list, filtered_word_labels
