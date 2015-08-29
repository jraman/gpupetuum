'''
Given input in LLC binary format, save as pickled file.

If selectlabelfile (and outlabelfile) is specified, only convert for those labels specified in selectlabelfile
and write out labels to outlabelfile.
If converting all samples, inlabelfile is redundant, but makes the code simpler.
'''

import argparse
import cPickle
import functools
import logging
import numpy as np
import struct


SIZE_FLOAT = 4
NUM_FEATURES_IMNET = 21504
NUM_FEATURES_TIMIT = 360


class Llc2Pickle(object):
    def __init__(self, infile, inlabelfile, num_features, outfile, selectlabelfile, outlabelfile):
        '''
        :type infile: string
        :param infile: input imnet file of binary data (float32)

        :type inlabelfile: string
        :param inlabelfile: input (full) label file.  Each each line has an int which is the label.

        :type outfile: string
        :param outfile: pickled feature vectors are saved as a stream
            of feature vectors.

        :type outlabelfile: string
        :param outlabelfile: if None, then convert all data.  Else, only convert for all the labels in the input file.
        '''
        self.infile = infile
        self.inlabelfile = inlabelfile
        self.num_features = num_features
        self.outfile = outfile
        self.selectlabelfile = selectlabelfile
        if selectlabelfile:
            self.outlabelfile = outlabelfile
        else:
            self.outlabelfile = '/dev/null'

        self.outlabels = set()
        logging.info('infile: {}, outfile: {}, inlabelfile: {}, selectlabelfile: {}, outlabelfile: {}'.format(
            self.infile, self.outfile, self.inlabelfile, self.selectlabelfile, self.outlabelfile))

    def process_file(self):
        '''Read binary imnet infile and write to outfile'''
        sample_output = self.selectlabelfile is not None
        if sample_output:
            self.outlabelset = self.read_outlabels()

        with open(self.infile, 'rb') as fin, open(self.inlabelfile, 'r') as finlab:
            with open(self.outfile, 'wb') as fout, open(self.outlabelfile, 'wb') as foutlab:
                end_of_label_file = False
                ii = 0
                while True:
                    try:
                        # read label regardless of sample_output, so we can check EOF of label and data files
                        label = int(finlab.next())
                    except StopIteration:
                        label = None
                        end_of_label_file = True
                    skip = sample_output and label not in self.outlabelset
                    sample = self.read_sample(fin, skip)
                    if end_of_label_file:
                        assert sample is None, 'Label file ended before data file'
                    if sample is None:
                        # EOF
                        assert end_of_label_file, 'Data file ended before label file'
                        break
                    if not skip:
                        cPickle.dump(sample, fout, protocol=2)
                        if sample_output:
                            foutlab.write('{}\n'.format(label))
                            logging.debug('wrote 1 sample for label {}'.format(label or 'next'))
                    ii += 1
                    if ii % 1000 == 0:
                        logging.debug('processed {}'.format(ii))

    def read_sample(self, fin, skip):
        '''Read file and yield one feature vector.
        Returns None at EOF.

        :type fin: file like object
        :param fin: input file

        :type skip: boolean
        :param skip: if True, read but don't convert.  Returns []

        :output: sample vector, or [] if skip, or None if EOF
        '''
        raw = fin.read(self.num_features * SIZE_FLOAT)
        if not raw:
            # at EOF, raw == ''
            return None
        assert len(raw) == self.num_features * SIZE_FLOAT, 'read incomplete vector at EOF'
        x = []
        if not skip:
            x = [y[0] for y in [struct.unpack('f', raw[i:(i + 4)]) for i in xrange(0, len(raw), 4)]]
            x = np.asarray(x, dtype=np.float32)
        return x

    def read_outlabels(self):
        with open(self.selectlabelfile) as flab:
            labels = set(int(x.strip()) for x in flab)
        return labels


def main():
    '''
    :param args[0]: input file
    :param args[1]: input label file
    :param args[2]: output file
    :param args[3]: (optional) output label selection file
    :param args[4]: (optional) output label file (required if selection file above is specified)
    '''
    args = _myargparse()
    log_format = '%(asctime)s %(name)s %(filename)s:%(lineno)d %(levelname)s %(message)s'
    logging.basicConfig(format=log_format, level=logging.DEBUG)
    converter = Llc2Pickle(args.infile, args.labelfile, args.num_features,
                           args.outfile, args.selectlabelfile, args.outlabelfile)
    converter.process_file()


def _myargparse():
    help_formatter = functools.partial(argparse.ArgumentDefaultsHelpFormatter, width=104)
    parser = argparse.ArgumentParser(
        description='Feature Extractor',
        formatter_class=help_formatter)
    parser.add_argument('-i', '--infile', action='store', help='input LLC binary file', required=True)
    parser.add_argument('-o', '--outfile', action='store', help='output pkl files', required=True)
    parser.add_argument('-n', '--num-features', action='store', type=int, help='feature dimension', required=True)
    parser.add_argument('-l', '--labelfile', action='store', help='label file', required=True)
    parser.add_argument('-s', '--selectlabelfile', action='store', help='file with a list of labels to select')
    parser.add_argument('--outlabelfile', action='store',
                        help='in the case of selectlabelfile, corresponding lablefile')
    args = parser.parse_args()
    if args.selectlabelfile:
        assert args.outlabelfile, 'If selectlabelfile, then outlabelfile is required'
    return args


if __name__ == '__main__':
    main()
