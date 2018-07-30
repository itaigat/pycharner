from os import rename, listdir, path


def zfill_files(folder):
    """
    This function takes a folder and standarts all the files to the same length
    for example:
    ['1.xml', '23.xml'] -> ['01.xml', 23.xml']
    :param folder: Folder you want to change
    """

    files = [f for f in listdir(folder)]
    maxlen = max([len(f) for f in files]) - 4
    for file in files:
        f_lst = file.split('.')
        rename(path.join(folder, file), path.join(folder, f_lst[0].zfill(maxlen) + '.' + f_lst[1]))
