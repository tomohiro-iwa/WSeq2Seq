
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('SOURCE0', help='source sentence list')
    parser.add_argument('SOURCE1', help='source sentence list')
    parser.add_argument('TARGET', help='target sentence list')
    parser.add_argument('SOURCE_VOCAB0', help='source vocabulary file')
    parser.add_argument('SOURCE_VOCAB1', help='source vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')
    parser.add_argument('--validation-source0',
                        help='source sentence list for validation')
    parser.add_argument('--validation-source1',
                        help='source sentence list for validation')
    parser.add_argument('--validation-target',
                        help='target sentence list for validation')
    parser.add_argument('--test-source0',
                        help='source sentence list for test')
    parser.add_argument('--test-source1',
                        help='source sentence list for test')
    parser.add_argument('--test-target',
                        help='target sentence list for test')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--resume-emb',  default='',
                        help='resume the training from snapshot')
    parser.add_argument('--resume-enco', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--save', '-s', default='',
                        help='save a snapshot of the training')
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='number of units')
    parser.add_argument('--l2', '-l2', type=float, default=0.001,
                        help='number of l2 arg')
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--use-dataset-api', default=False,
                        action='store_true',
                        help='use TextDataset API to reduce CPU memory usage')
    parser.add_argument('--min-source-sentence', type=int, default=1,
                        help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=50,
                        help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=1,
                        help='minimium length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=50,
                        help='maximum length of target sentence')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='number of iteration to show log')
    parser.add_argument('--validation-interval', type=int, default=200,
                        help='number of iteration to evlauate the model '
                        'with validation dataset')
    parser.add_argument('--test-interval', type=int, default=200,
                        help='number of iteration to evlauate the model '
                        'with test dataset')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    return parser
	
