import os
import time
import json
import subprocess
import argparse
import bisect
import random
import mxnet as mx
import numpy as np

parser = argparse.ArgumentParser(description="Run Inference using Pen Tree Bank Dataset",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpus', type=int,
                    help='Number of GPUs to use, 0 means using CPU only.')


def pad_sentence(sentence, buckets, invalid_label=-1, data_name='data', layout='NT'):
    """
    Pad a sentence to closest length in provided buckets.

    :param sentence: list of int
        A list of integer representing an encoded sentence.
    :param buckets: list of int
        Size of the data buckets.
    :param invalid_label: int, optional
        Index for invalid token, like <end-of-sentence>.
    :param data_name: str, optional
        Input data name.
    :param layout: str, optional
        Format of data and label. 'NT' means (batch_size, length)
        and 'TN' means (length, batch_size).

    :return: mx.io.DataBatch
        DataBatch contains sentence.
    """
    if len(sentence) > 60:
        sentence = sentence[:60]
    buck = bisect.bisect_left(buckets, len(sentence))
    buff = np.full((buckets[buck],), invalid_label, dtype='float32')
    buff[:len(sentence)] = sentence
    sent_bucket = buckets[buck]
    pad_sent = mx.nd.array([buff], dtype='float32')
    shape = (1, sent_bucket) if layout == 'NT' else (sent_bucket, 1)
    return mx.io.DataBatch([pad_sent], pad=0, bucket_key=sent_bucket,
                           provide_data=[mx.io.DataDesc(
                               name=data_name,
                               shape=shape,
                               layout=layout)])




def tokenize_text(fname, vocab=None, invalid_label=-1, start_label=0):
    try:
        assert os.path.exists(fname)
    except AssertionError:
        subprocess.call(['{}/get_ptb_data.sh'.format(os.path.dirname(__file__))])
        assert os.path.exists(fname)
    lines = open(fname).readlines()
    lines = [filter(None, i.split(' ')) for i in lines]
    sentences, vocab = mx.rnn.encode_sentences(lines, vocab=vocab, invalid_label=invalid_label,
                                               start_label=start_label)
    return sentences, vocab

def sym_gen(seq_len):
    data = mx.sym.Variable("data")
    embed = mx.sym.Embedding(data=data, input_dim=len(vocab),
                             output_dim=num_embed, name="embed")
    stack.reset()
    outputs, _ = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

    pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
    pred = mx.sym.FullyConnected(data=pred, num_hidden=len(vocab), name="pred")
    pred = mx.sym.softmax(pred, name='softmax')
    return pred, ('data',), ('softmax_label',)

if __name__ == '__main__':
    args = parser.parse_args()

    model_dir = "model"

    buckets = [10, 20, 30, 40, 50, 60]
    start_label = 1
    invalid_key = "\n"
    invalid_label = 0
    vocab = {}
    idx2word = {}
    epoch = 100

    mx.test_utils.download("https://s3.amazonaws.com/model-server/models/lstm_ptb/lstm_ptb-symbol.json", dirname=model_dir)
    mx.test_utils.download("https://s3.amazonaws.com/model-server/models/lstm_ptb/lstm_ptb-0100.params", dirname=model_dir)
    mx.test_utils.download("https://s3.amazonaws.com/model-server/models/lstm_ptb/vocab_dict.txt", dirname=model_dir)
    mx.test_utils.download("https://s3.amazonaws.com/model-server/models/lstm_ptb/signature.json", dirname=model_dir)

    signature_file_path = os.path.join(model_dir, "signature.json")
    if not os.path.isfile(signature_file_path):
        raise RuntimeError("Missing signature.json file.")

    with open(signature_file_path) as f:
        signature = json.load(f)

    data_names = []
    data_shapes = []

    for input_data in signature["inputs"]:
        data_names.append(input_data["data_name"])
        data_shapes.append((input_data["data_name"], tuple(input_data["data_shape"])))

    vocab_dict_file = os.path.join(model_dir, "vocab_dict.txt")

    with open(vocab_dict_file, 'r') as vocab_file:
        vocab[invalid_key] = invalid_label
        for line in vocab_file:
            word_index = line.split(' ')
            if len(word_index) < 2 or word_index[0] == '':
                continue
            vocab[word_index[0]] = int(word_index[1].rstrip())
    for key, val in vocab.items():
        idx2word[val] = key

    # Load pre-trained lstm bucketing module
    num_layers = 2
    num_hidden = 200
    num_embed = 200

    stack = mx.rnn.FusedRNNCell(num_hidden, num_layers=num_layers, mode="lstm").unfuse()

    mxnet_ctx = mx.cpu()

    # Create bucketing module and load weights
    mx_model = mx.mod.BucketingModule(
            sym_gen=sym_gen,
            default_bucket_key=max(buckets),
            context=mxnet_ctx)

    checkpoint_prefix = "{}/{}".format(model_dir, "lstm_ptb")

    mx_model.bind(data_shapes=data_shapes, for_training=False)

    _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(stack, checkpoint_prefix, epoch)
    mx_model.set_params(arg_params, aux_params)
    val_sent, _ = tokenize_text("{}/data/ptb.test.txt".format(os.path.dirname(__file__)),
                                vocab=vocab,
                                start_label=start_label,
                                invalid_label=invalid_label)

    for val in val_sent:
        data_batch = pad_sentence(val, buckets, invalid_label=invalid_label, data_name=data_names[0], layout="NT")
        start = time.time()
        mx_model.forward(data_batch)
        mx_model.get_outputs()[0].wait_to_read()
        end = time.time() - start
        print("Infer time is {}".format(end))
