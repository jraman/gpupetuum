import cPickle


def load_pickle_file(fobj):
    while True:
        try:
            yield cPickle.load(fobj)
        except EOFError:
            break
