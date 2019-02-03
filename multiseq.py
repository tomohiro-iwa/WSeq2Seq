import argparse
import datetime
from logzero import logger

from nltk.translate import bleu_score
import numpy
import progressbar
import six

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import random


UNK = 0
EOS = 1


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class Multiseq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source0_vocab, n_source1_vocab, n_target_vocab, n_units):
        super(Multiseq2seq, self).__init__()
        with self.init_scope():
            self.embed_x0 = L.EmbedID(n_source0_vocab, n_units)
            self.embed_x1 = L.EmbedID(n_source1_vocab, n_units)
            self.embed_y = L.EmbedID(n_target_vocab, n_units*2)
            self.encoder0 = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.encoder1 = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.decoder = L.NStepLSTM(n_layers, n_units*2, n_units*2, 0.1)
            self.W = L.Linear(n_units*2, n_target_vocab)

        self.n_layers = n_layers
        self.n_units = n_units

    def forward(self, xs0, xs1, ys):
        xs0 = [x[::-1] for x in xs0]
        xs1 = [x[::-1] for x in xs1]

        eos = self.xp.array([EOS], numpy.int32)
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # Both xs and ys_in are lists of arrays.
        exs0 = sequence_embed(self.embed_x0, xs0)
        exs1 = sequence_embed(self.embed_x1, xs1)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs0)
        # None represents a zero vector in an encoder.
        hx0, cx0, _ = self.encoder0(None, None, exs0)
        hx1, cx1, _ = self.encoder1(None, None, exs1)
        hx = F.concat([hx0, hx1], axis=2)
        cx = F.concat([cx0, cx1], axis=2)
        _, _, os = self.decoder(hx, cx, eys)

        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, reduce='no')) / batch

        chainer.report({'loss': loss}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.array * batch / n_words)
        chainer.report({'perp': perp}, self)
        return loss

    def translate(self, xs0, xs1, max_length=100):
        batch = len(xs0)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs0 = [x[::-1] for x in xs0]
            xs1 = [x[::-1] for x in xs1]
            exs0 = sequence_embed(self.embed_x0, xs0)
            exs1 = sequence_embed(self.embed_x1, xs1)
            h0, c0, _ = self.encoder0(None, None, exs0)
            h1, c1, _ = self.encoder1(None, None, exs1)
            h = F.concat([h0, h1], axis=2)
            c = F.concat([c0, c1], axis=2)
            ys = self.xp.full(batch, EOS, numpy.int32)
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.array, axis=1).astype(numpy.int32)
                result.append(ys)

        # Using `xp.concatenate(...)` instead of `xp.stack(result)` here to
        # support NumPy 1.9.
        result = cuda.to_cpu(
            self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs


def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x)
                                     for x in batch[:-1]], dtype=numpy.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'xs0': to_device_batch([x for x, _, _ in batch]),
            'xs1': to_device_batch([x for _, x, _ in batch]),
            'ys': to_device_batch([y for _, _, y in batch])}


class CalculateBleu(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, valid_data, key, batch=100, device=-1, max_length=100):
        self.model = model
        self.valid_data = valid_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.valid_data), self.batch):
#               sources, targets = zip(*self.valid_data[i:i + self.batch])
                sources0, sources1, targets = zip(*self.valid_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])

#                sources = [
#                    chainer.dataset.to_device(self.device, x) for x in sources]
                sources0 = [
                    chainer.dataset.to_device(self.device, x) for x in sources0]
                sources1 = [
                    chainer.dataset.to_device(self.device, x) for x in sources1]
#                ys = [y.tolist()
#                      for y in self.model.translate(sources, self.max_length)]
                ys = [y.tolist()
                      for y in self.model.translate(sources0, sources1, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        chainer.report({self.key: bleu})

    def forward(self, trainer):
        raise NotImplementedError('must not happen.')


def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with open(path) as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = 0
    word_ids['<EOS>'] = 1
    return word_ids


def load_data(vocabulary, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    data = []
    logger.info(f'loading {path} ...')
    with open(path) as f:
        for line in bar(f, max_value=n_lines):
            words = line.strip().split()
            array = numpy.array([vocabulary.get(w, UNK)
                                 for w in words], numpy.int32)
            data.append(array)
    return data


#def load_data_using_dataset_api(
#        src_vocab, src_path, target_vocab, target_path, filter_func):
#
#    def _transform_line(vocabulary, line):
#        words = line.strip().split()
#        return numpy.array(
#            [vocabulary.get(w, UNK) for w in words], numpy.int32)
#
#    def _transform(example):
#        source, target = example
#        return (
#            _transform_line(src_vocab, source),
#            _transform_line(target_vocab, target)
#        )
#
#    return chainer.datasets.TransformDataset(
#        chainer.datasets.TextDataset(
#            [src_path, target_path],
#            encoding='utf-8',
#            filter_func=filter_func
#        ), _transform)


def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total


def generate_argparser():
    parser = argparse.ArgumentParser(description='Multiple sequence to sequence',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('SOURCE0', help='source0 sentence list')
    parser.add_argument('SOURCE1', help='source1 sentence list')
    parser.add_argument('TARGET', help='target sentence list')

    parser.add_argument('SOURCE0_VOCAB', help='source0 vocabulary file')
    parser.add_argument('SOURCE1_VOCAB', help='source1 vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')

    parser.add_argument('--dev-source0',
                        help='source0 sentence list of development dataset')
    parser.add_argument('--dev-source1',
                        help='source1 sentence list of development dataset')
    parser.add_argument('--dev-target',
                        help='target sentence list of development dataset')

    parser.add_argument('--test-source0',
                        help='source0 sentence list for test')
    parser.add_argument('--test-source1',
                        help='source1 sentence list for test')
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
    parser.add_argument('--save', '-s', default='',
                        help='save a snapshot of the training')
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='number of units')
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
#    parser.add_argument('--use-dataset-api', default=False,
#                        action='store_true',
#                        help='use TextDataset API to reduce CPU memory usage')
#    parser.add_argument('--min-source-sentence', type=int, default=1,
#                        help='minimium length of source sentence')
#    parser.add_argument('--max-source-sentence', type=int, default=50,
#                        help='maximum length of source sentence')
#    parser.add_argument('--min-target-sentence', type=int, default=1,
#                        help='minimium length of target sentence')
#    parser.add_argument('--max-target-sentence', type=int, default=50,
#                        help='maximum length of target sentence')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='number of iteration to show log')
    parser.add_argument('--validation-interval', type=int, default=4000,
                        help='number of iteration to evlauate the model '
                        'with development/test dataset')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    return parser

def reset_seed(seed=0):
    random.seed(seed)
    numpy.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)

def main():
    parser = generate_argparser()
    args = parser.parse_args()

    # Initialize random seed
    logger.info(f'initializing random seed by {args.seed} ...')
    reset_seed(args.seed)
    
    # Load vocabularies
    logger.info('loading vocabularies...')
    source0_ids = load_vocabulary(args.SOURCE0_VOCAB)
    source1_ids = load_vocabulary(args.SOURCE1_VOCAB)
    target_ids = load_vocabulary(args.TARGET_VOCAB)

    # Load all records on memory
    train_source0 = load_data(source0_ids, args.SOURCE0)
    train_source1 = load_data(source1_ids, args.SOURCE1)
    train_target = load_data(target_ids, args.TARGET)
    assert len(train_source0) == len(train_source1) == len(train_target)

    train_data = [
        (s0, s1, t)
        for s0, s1, t in six.moves.zip(train_source0, train_source1, train_target)
    ]
    logger.info('dataset loaded.')

    # Print dataset statistics
    train_source0_unknown = calculate_unknown_ratio(
        [s0 for s0, _, _ in train_data])
    train_source1_unknown = calculate_unknown_ratio(
        [s1 for _, s1, _ in train_data])
    train_target_unknown = calculate_unknown_ratio(
        [t for _, _, t in train_data])
    logger.info('Source0 vocabulary size: %d' % len(source0_ids))
    logger.info('Source1 vocabulary size: %d' % len(source1_ids))
    logger.info('Target vocabulary size: %d' % len(target_ids))
    logger.info('Train data size: %d' % len(train_data))
    logger.info('Train source0 unknown ratio: %.2f%%' % (
        train_source0_unknown * 100))
    logger.info('Train source1 unknown ratio: %.2f%%' % (
        train_source1_unknown * 100))
    logger.info('Train target unknown ratio: %.2f%%' % (
        train_target_unknown * 100))

    source0_words = {i: w for w, i in source0_ids.items()}
    source1_words = {i: w for w, i in source1_ids.items()}
    target_words = {i: w for w, i in target_ids.items()}

    # Setup model
    model = Multiseq2seq(args.layer, len(source0_ids), len(source1_ids), len(target_ids), args.unit)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    # Setup optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Setup iterator
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)

    # Setup updater and trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.LogReport(
        trigger=(args.log_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'dev/main/loss', 'test/main/loss',
         'main/perp', 'dev/main/perp', 'test/main/perp',
         'dev/main/bleu', 'test/main/bleu',
         'elapsed_time']),
        trigger=(args.log_interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(training_length=(args.epoch, 'epoch'), update_interval=1))

    # Setup extensions for validation (with DEVELOPMENT dataset)
    if args.dev_source0 and args.dev_source1 and args.dev_target:
        dev_source0 = load_data(source0_ids, args.dev_source0)
        dev_source1 = load_data(source1_ids, args.dev_source1)
        dev_target = load_data(target_ids, args.dev_target)
        assert len(dev_source0) == len(dev_source1) == len(dev_target)
        dev_data = list(six.moves.zip(dev_source0, dev_source1, dev_target))
        dev_data = [(s0, s1, t) for s0, s1, t in dev_data if 0 < len(s0) and 0 < len(s1) and 0 < len(t)]

        dev_source0_unknown = calculate_unknown_ratio(
            [s0 for s0, _, _ in dev_data])
        dev_source1_unknown = calculate_unknown_ratio(
            [s1 for _, s1, _ in dev_data])
        dev_target_unknown = calculate_unknown_ratio(
            [t for _, _, t in dev_data])
        logger.info('Dev data size: %d' % len(dev_data))
        logger.info('Dev source0 unknown ratio: %.2f%%' %
              (dev_source0_unknown * 100))
        logger.info('Dev source1 unknown ratio: %.2f%%' %
              (dev_source1_unknown * 100))
        logger.info('Dev target unknown ratio: %.2f%%' %
              (dev_target_unknown * 100))

        @chainer.training.make_extension()
        def translate(trainer):
            source0, source1, target = dev_data[numpy.random.choice(len(dev_data))]
            result = model.translate([model.xp.array(source0)], [model.xp.array(source1)])[0]

            source0_sentence = ' '.join([source0_words[x] for x in source0])
            source1_sentence = ' '.join([source1_words[x] for x in source1])
            target_sentence = ' '.join([target_words[y] for y in target])
            result_sentence = ' '.join([target_words[y] for y in result])
            logger.info('# source0 : ' + source0_sentence)
            logger.info('# source1 : ' + source1_sentence)
            logger.info('# result : ' + result_sentence)
            logger.info('# expect : ' + target_sentence)

        trainer.extend(
            translate, trigger=(args.validation_interval, 'iteration'))
        trainer.extend(
            CalculateBleu(model, dev_data, 'dev/main/bleu', device=args.gpu),
            trigger=(args.validation_interval, 'iteration'))

        dev_iter = chainer.iterators.SerialIterator(dev_data, args.batchsize, repeat=False, shuffle=False)
        dev_eval = extensions.Evaluator(dev_iter, model, device=args.gpu, converter=convert)
        dev_eval.name = 'dev'
        trainer.extend(dev_eval, trigger=(args.validation_interval, 'iteration'))

    # Setup extensions for validation (with TEST dataset)
    if args.test_source0 and args.test_source1 and args.test_target:
        test_source0 = load_data(source0_ids, args.test_source0)
        test_source1 = load_data(source1_ids, args.test_source1)
        test_target = load_data(target_ids, args.test_target)
        assert len(test_source0) == len(test_source1) == len(test_target)
        test_data = list(six.moves.zip(test_source0, test_source1, test_target))
        test_data = [(s0, s1, t) for s0, s1, t in test_data if 0 < len(s0) and 0 < len(s1) and 0 < len(t)]

        test_source0_unknown = calculate_unknown_ratio(
            [s0 for s0, _, _ in test_data])
        test_source1_unknown = calculate_unknown_ratio(
            [s1 for _, s1, _ in test_data])
        test_target_unknown = calculate_unknown_ratio(
            [t for _, _, t in test_data])
        logger.info('Test data size: %d' % len(test_data))
        logger.info('Test source0 unknown ratio: %.2f%%' %
              (test_source0_unknown * 100))
        logger.info('Test source1 unknown ratio: %.2f%%' %
              (test_source1_unknown * 100))
        logger.info('Test target unknown ratio: %.2f%%' %
              (test_target_unknown * 100))

        trainer.extend(
            CalculateBleu(model, test_data, 'test/main/bleu', device=args.gpu),
            trigger=(args.validation_interval, 'iteration'))

        test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize, repeat=False, shuffle=False)
        test_eval = extensions.Evaluator(test_iter, model, device=args.gpu, converter=convert)
        test_eval.name = 'test'
        trainer.extend(test_eval, trigger=(args.validation_interval, 'iteration'))

    if args.resume:
        # Resume from a snapshot
        logger.info(f'loading model {args.resume} ...')
        chainer.serializers.load_npz(args.resume, trainer)

    logger.info('start training...')
    trainer.run()

    if args.save:
        # Save a snapshot
        logger.info(f'saving model {args.save} ...')
        chainer.serializers.save_npz(args.save, trainer)


if __name__ == '__main__':
    main()
