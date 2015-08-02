'''
Input file:
 * gzipped
 * stream of ndarrays.  All arrays of equal length.
 * Each array is one feature vector.
'''

import cPickle as pickle
import gzip
import sys


filename = sys.argv[1]

ref_veclen = None
with gzip.open(filename) as fin:
    ii = 0
    while True:
        try:
            fvec = pickle.load(fin)
        except EOFError:
            break
        ref_veclen = ref_veclen or len(fvec)
        assert ref_veclen == len(fvec), 'inconsistent vector length: found {}, expected {}, vector num: {}'.format(
            ref_veclen, len(fvec), ii)
        ii += 1

print 'file {} has {} vectors of length {}'.format(filename, ii, ref_veclen)
