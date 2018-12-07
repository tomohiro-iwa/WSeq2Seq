#!/usr/bin/python3

import argparse
from arg_parser import get_parser
import datetime

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

from chainerui.utils import save_args

import random
UNK = 0
EOS = 1


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    try:
        ex = embed(F.concat(xs, axis=0))
    except:
        print(xs)
    exs = F.split_axis(ex, x_section, 0)
    return exs


class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source0_vocab,n_source1_vocab, n_target_vocab, n_units):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.embed_x0 = L.EmbedID(n_source0_vocab, n_units)
            self.embed_x1 = L.EmbedID(n_source1_vocab, n_units)
            self.embed_y = L.EmbedID(n_target_vocab, n_units)
            self.encoder0 = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.encoder1 = L.NStepLSTM(n_layers, n_units, n_units, 0.1)

            self.decoder = L.NStepLSTM(n_layers, n_units*2, n_units*2, 0.1)
            self.W = L.Linear(n_units*2, n_target_vocab)

        self.n_layers = n_layers
        self.n_units = n_units

    def forward(self, xs0,xs1, ys):
        xs0 = [x[::-1] for x in xs0]
        xs1 = [x[::-1] for x in xs1]

        eos = self.xp.array([EOS], numpy.int32)
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # Both xs and ys_in are lists of arrays.
        exs0 = sequence_embed(self.embed_x0, xs0)
        exs1 = sequence_embed(self.embed_x1, xs1)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs1)
        # None represents a zero vector in an encoder.
        hx0, cx0, _ = self.encoder0(None, None, exs0)
        hx1, cx1, _ = self.encoder1(None, None, exs1)
 
        hx = F.concat([hx0,hx1],axis=2)
        cx = F.concat([cx0,cx1],axis=2)

        #改造　ここまで
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
            
            hx0, cx0, _ = self.encoder0(None, None, exs0)
            hx1, cx1, _ = self.encoder1(None, None, exs1)
            
            h = F.concat([hx0,hx1],axis=2)
            c = F.concat([cx0,cx1],axis=2)

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
            self, model, test_data, key, batch=100, device=-1, max_length=200):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources0,sources1, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])

                sources0 = [
                    chainer.dataset.to_device(self.device, x) for x in sources0]
                sources1 = [
                    chainer.dataset.to_device(self.device, x) for x in sources1]
                ys = [y.tolist()
                      for y in self.model.translate(sources0,sources1, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        chainer.report({self.key: bleu})


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
    print('loading...: %s' % path)
    with open(path) as f:
        for line in bar(f, max_value=n_lines):
            words = line.strip().split()
            array = numpy.array([vocabulary.get(w, UNK)
                                 for w in words], numpy.int32)
            data.append(array)
    return data

def make_data_tuple(source0=(),source1=(),target=()):
    # Load all records on memory.
    source0_data = load_data(source0[0], source0[1])
    source1_data = load_data(source1[0], source1[1])
    target_data = load_data(target[0], target[1])
    assert len(source0_data) == len(source1_data) == len(target_data)
    
    #長さのフィルターを削除
    data = [
        (s0,s1, t)
        for s0,s1, t in six.moves.zip(source0_data,source1_data, target_data)
    ]
    source0_unknown = calculate_unknown_ratio(
        [s for s, _, _ in data])
    source1_unknown = calculate_unknown_ratio(
        [s for _, s, _ in data])
    target_unknown = calculate_unknown_ratio(
        [t for _, _, t in data])

    print('data len: %d' % len(data))
    print('source0 unknown ratio: %.2f%%' %
          (source0_unknown * 100))
    print('source1 unknown ratio: %.2f%%' %
          (source1_unknown * 100))
    print('target unknown ratio: %.2f%%' %
          (target_unknown * 100))
    return data

def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total

def reset_seed(seed=0):
    random.seed(seed)
    numpy.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


def main():

    chainer.set_debug(True)
    parser = get_parser()
    args = parser.parse_args()

    reset_seed(args.seed)

    #load vocabulary
    source0_ids = load_vocabulary(args.SOURCE_VOCAB0)
    source1_ids = load_vocabulary(args.SOURCE_VOCAB1)
    target_ids = load_vocabulary(args.TARGET_VOCAB)
    print('Source vocabulary size: %d' % len(source0_ids))
    print('Source vocabulary size: %d' % len(source1_ids))
    print('Target vocabulary size: %d' % len(target_ids))

    train_data = make_data_tuple(
        source0 = (source0_ids, args.SOURCE0),
        source1 = (source1_ids, args.SOURCE1),
        target = (target_ids, args.TARGET)
    )

    source0_words = {i: w for w, i in source0_ids.items()}
    source1_words = {i: w for w, i in source1_ids.items()}
    target_words = {i: w for w, i in target_ids.items()}

    # Setup model
    model = Seq2seq(args.layer, len(source0_ids),len(source1_ids), len(target_ids), args.unit)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    # Setup optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.l2))

    # Setup iterator
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)

    # Setup updater and trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.LogReport(
        trigger=(args.log_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
         'main/perp', 'validation/main/perp', 'validation/main/bleu',
         'test/main/bleu','elapsed_time']),
        trigger=(args.log_interval, 'iteration'))

    if args.validation_source0 and args.validation_source1 and args.validation_target:
        valid_data = make_data_tuple(
            source0 = (source0_ids, args.validation_source0),
            source1 = (source1_ids, args.validation_source1),
            target = (target_ids, args.validation_target)
        )

        @chainer.training.make_extension()
        def translate(trainer):
            source0, source1 , target = valid_data[numpy.random.choice(len(valid_data))]
            result = model.translate([model.xp.array(source0)], [model.xp.array(source1)])[0]

            source0_sentence = ' '.join([source0_words[x] for x in source0])
            source1_sentence = ' '.join([source1_words[x] for x in source1])
            target_sentence = ' '.join([target_words[y] for y in target])
            result_sentence = ' '.join([target_words[y] for y in result])
            print('# source0 : ' + source0_sentence)
            print('# source1 : ' + source1_sentence)
            print('# result : ' + result_sentence)
            print('# expect : ' + target_sentence)

        trainer.extend(
            translate, trigger=(args.validation_interval, 'iteration'))
        trainer.extend(
            CalculateBleu(
                model, valid_data, 'validation/main/bleu', device=args.gpu),
            trigger=(args.validation_interval, 'iteration'))

        dev_iter = chainer.iterators.SerialIterator(valid_data, args.batchsize, repeat=False, shuffle=False)
        dev_eval = extensions.Evaluator(dev_iter, model, device=args.gpu, converter=convert)
        dev_eval.name = 'valid'
        trainer.extend(dev_eval, trigger=(args.validation_interval, 'iteration'))

    if args.test_source0 and args.test_source1 and args.test_target:
        test_data = make_data_tuple(
            source0 = (source0_ids, args.test_source0),
            source1 = (source1_ids, args.test_source1),
            target = (target_ids, args.test_target)
        )
        trainer.extend(
            CalculateBleu(
                model, test_data, 'test/main/bleu', device=args.gpu),
            trigger=(args.test_interval, 'iteration'))

    print('start training')
    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)
    save_args(args, args.out)

    trainer.run()

    if args.save:
        # Save a snapshot
        chainer.serializers.save_npz(args.out+"/trainer.npz", trainer)
        chainer.serializers.save_npz(args.out+"/model.npz", model)


if __name__ == '__main__':
    main()
