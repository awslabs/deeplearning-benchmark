# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import time
import math
import mxnet as mx
from mxnet import gluon, autograd
import model
import data
import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='MXNet Autograd PennTreeBank RNN/LSTM Language Model')

parser.add_argument('--data', type=str, default='./data/ptb.',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='lstm',
                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1.0,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size per device')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.params',
                    help='path to save the final model')
parser.add_argument('--mode', type=str, default='imperative',
                    help='mode of the gluon rnn model. (imperative, hybrid)')
parser.add_argument('--kvstore', type=str, default='device',help='kvstore to use for trainer/module.')
parser.add_argument('--dtype', type=str, default='float32',help='floating point precision to use')
args = parser.parse_args()


def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data


def get_batch(source, i):
    seq_len = min(args.bptt, source.shape[0] - 1 - i)
    data = source[i:i+seq_len] # input
    target = source[i+1:i+1+seq_len] # label
    return data, target


def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


def eval(data_source, ctx):
    total_L = 0.0
    ntotal = 0
    hidden_states = [
        model.begin_state(func=mx.nd.zeros, batch_size=args.batch_size/len(ctx), ctx=ctx[i])
        for i in range(len(ctx))
    ]
    for i in range(0, data_source.shape[0] - 1, args.bptt):
        data_batch, target_batch = get_batch(data_source, i)
        data = gluon.utils.split_and_load(data_batch, ctx_list=ctx, batch_axis=1)
        target = gluon.utils.split_and_load(target_batch, ctx_list=ctx, batch_axis=1)
        for (d, t) in zip(data, target):
            hidden = hidden_states[d.context.device_id]
            output, hidden = model(d, hidden)
            L = loss(output, t.reshape((-1,)))
            total_L += mx.nd.sum(L).asscalar()
            ntotal += L.size
    return total_L / ntotal


###############################################################################
# Load data and Build the model
###############################################################################

if args.gpus > 0:
    context = [mx.gpu(i) for i in range(args.gpus)]
else:
    context = [mx.cpu(0)]

corpus = data.Corpus(args.data)

args.batch_size *= max(1, args.gpus)
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, args.batch_size)
test_data = batchify(corpus.test, args.batch_size)
n_tokens = len(corpus.dictionary)


model = model.RNNModel(args.model, n_tokens, args.emsize, args.nhid,
                       args.nlayers, args.dropout, args.tied)

model.collect_params().initialize(mx.init.Xavier(), ctx=context)

trainer = gluon.Trainer(
    model.collect_params(), 'sgd',
    {
        'learning_rate': args.lr,
        'momentum': 0,
        'wd': 0
    }
)
loss = gluon.loss.SoftmaxCrossEntropyLoss()


###############################################################################
# Train the model
###############################################################################

def train(epochs, ctx):
    best_val = float("Inf")

    for epoch in range(epochs):
        total_L = 0.0
        start_time = time.time()
        hidden_states = [
            model.begin_state(func=mx.nd.zeros, batch_size=args.batch_size // len(ctx), ctx=ctx[i])
            for i in range(len(ctx))
        ]
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, args.bptt)):
            # get data batch from the training data
            data_batch, target_batch = get_batch(train_data, i)
            # For RNN we can do within batch multi-device parallelization
            data = gluon.utils.split_and_load(data_batch, ctx_list=ctx, batch_axis=1)
            target = gluon.utils.split_and_load(target_batch, ctx_list=ctx, batch_axis=1)
            Ls = []
            for (d, t) in zip(data, target):
                # get corresponding hidden state then update hidden
                hidden = detach(hidden_states[d.context.device_id])
                with autograd.record():
                    output, hidden = model(d, hidden)
                    L = loss(output, t.reshape((-1,)))
                    L.backward()
                    Ls.append(L)
                # write back to the record
                hidden_states[d.context.device_id] = hidden

            for c in ctx:
                grads = [i.grad(c) for i in model.collect_params().values()]
                # Here gradient is for the whole batch.
                # So we multiply max_norm by batch_size and bptt size to balance it.
                # Also this utility function needs to be applied within the same context
                gluon.utils.clip_global_norm(grads, args.clip * args.bptt * args.batch_size / len(ctx))

            trainer.step(args.batch_size)
            for L in Ls:
                total_L += mx.nd.sum(L).asscalar()

            if ibatch % args.log_interval == 0 and ibatch > 0:
                cur_L = total_L / args.bptt / args.batch_size / args.log_interval
                logging.info('[Epoch %d Batch %d] loss %.2f, ppl %.2f' % (
                    epoch, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0

        val_L = eval(val_data, ctx)

        logging.info('[Epoch %d] time cost %.2fs, valid loss %.2f, valid ppl %.2f' % (
            epoch, time.time() - start_time, val_L, math.exp(val_L)))

        if val_L < best_val:
            best_val = val_L
            test_L = eval(test_data, ctx)
            model.collect_params().save('model.params')
            logging.info('test loss %.2f, test ppl %.2f' % (test_L, math.exp(test_L)))
        else:
            args.lr = args.lr * 0.25
            trainer._init_optimizer('sgd',
                {
                    'learning_rate': args.lr,
                    'momentum': 0,
                    'wd': 0
                }
            )
            model.collect_params().load('model.params', ctx)


if __name__ == '__main__':
    if args.mode == 'hybrid':
        model.hybridize()

    ###############################################################################
    # Training code
    ###############################################################################
    train(args.epochs, context)
    model.collect_params().load(args.save, context)
    test_L = eval(test_data, context)
    logging.info('Best test loss %.2f, test ppl %.2f' % (test_L, math.exp(test_L)))
