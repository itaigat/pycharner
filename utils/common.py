from os import rename, listdir, path

from models.paramaters import DatasetsPaths
from utils.decoder import SportDataset


def zfill_files(folder):
    """
    This function takes a folder and standards all the files to the same length
    For example:
    ['1.xml', '23.xml'] -> ['01.xml', 23.xml']
    :param folder: Path to the folder you wish to change
    """

    files = [f for f in listdir(folder)]
    maxlen = max([len(f) for f in files]) - 4
    for file in files:
        f_lst = file.split('.')
        rename(path.join(folder, file), path.join(folder, str(f_lst[0].zfill(maxlen)) + '.' + str(f_lst[1])))


def print_sport_statistics(part='train'):
    """
    Location contains (Location, town, country)
    Person contains (Person)
    Organization contains (Organization)
    Misc contains (dateTime, numeral cardinal, product, language, time)
    :param part: The part we want to get statistics on
    """
    if part != 'train' and part != 'test' and part != 'valid':
        raise ValueError

    sport = SportDataset(DatasetsPaths.Sport5, part=part)
    words_counter = 0
    articles_counter = 0
    tags_counter = {}
    for words, tags, features in sport:
        words_counter += len(words)
        articles_counter += 1
        for tag in tags:
            if tag != 'None':
                if tag in tags_counter.keys():
                    tags_counter[tag] += 1
                else:
                    tags_counter[tag] = 1

    print('Number of words: ', words_counter)
    print('Number of Articles: ', articles_counter)
    print('Number of Locations: ', tags_counter['location'] + tags_counter['town'] + tags_counter['country'])
    print('Number of Persons: ', tags_counter['person'])
    print('Number of MISC: ', tags_counter['dateTime'] + tags_counter['numeral cardinal'] + tags_counter['product'] +
          tags_counter['language'] + tags_counter['time'])
    print('Number of Organization: ', tags_counter['organization'])
