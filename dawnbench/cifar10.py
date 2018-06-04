from __future__ import division

import argparse
import logging
from matplotlib import pyplot as plt
import mxnet as mx
import os
import time


class CIFAR10():
    def __init__(self, batch_size, data_shape, resize=-1):
        self.download()
        self.prepare_iters(batch_size, data_shape, resize)

    def download(self):
        parent_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
        self.data_path = data_path = os.path.join(parent_path, "data")
        cifar_path = os.path.join(data_path, "cifar")

        if not os.path.isdir(data_path):
            os.system("mkdir " + data_path)
        if (not os.path.exists(os.path.join(cifar_path, 'train.rec'))) or \
                (not os.path.exists(os.path.join(cifar_path, 'test.rec'))) or \
                (not os.path.exists(os.path.join(cifar_path, 'train.lst'))) or \
                (not os.path.exists(os.path.join(cifar_path, 'test.lst'))):
            print("Couldn't find CIFAR10 data, downloading...")
            os.system("wget -q http://data.mxnet.io/mxnet/data/cifar10.zip -P " + data_path)
            print("Download complete.")
            os.system("unzip -u " + os.path.join(data_path, "cifar10.zip") + " -d " + data_path)

    def prepare_iters(self, batch_size, data_shape, resize):
        """
        :param resize: if set to -1, this will not reshape the input data
        :return: a pair of iterators
        """
        self.train_iter_args = {
            'path_imgrec': os.path.join(self.data_path, "cifar/train.rec"),
            'data_shape': data_shape,
            'batch_size': batch_size,
            'shuffle': True,
            'resize': resize,
            'rand_crop': True,
            'rand_mirror': True,
            'mean_img': os.path.join(self.data_path, "cifar/mean.bin"),
            'pad': 12,
            'fill_value': 0,
            'max_random_illumination': 20,
        }
        self.train_iter = mx.io.ImageRecordIter(**self.train_iter_args)
        self.test_iter_args = {
            'path_imgrec': os.path.join(self.data_path, "cifar/test.rec"),
            'data_shape': data_shape,
            'batch_size': batch_size,
            'resize': resize,
            'rand_crop': False,
            'rand_mirror': False,
            'mean_img': os.path.join(self.data_path, "cifar/mean.bin"),
        }
        self.test_iter = mx.io.ImageRecordIter(**self.test_iter_args)

    def return_iters(self):
        return self.train_iter, self.test_iter

    def inspect_iter(self, stage="train", n_samples=5, idx=0):
        """
        :param stage: str, 'train' or 'test'
        :param n_samples: number of images to sample
        :param idx: int, index of image to slice out of batch
        :return: None, just plots
        """
        iter_args = dict(getattr(self, stage + "_iter_args")) # take copy using dict
        iter_args['shuffle'] = False
        sample_iter = mx.io.ImageRecordIter(**iter_args)
        for i in range(n_samples):
            sample_iter.reset()
            sample_batch = sample_iter.__next__().data[0]
            sample_image = sample_batch[idx].transpose((1,2,0)).asnumpy()
            plt.imshow((sample_image/255)+0.5)
            plt.show()


class ResnetModel():
    def __init__(self):
        # make into a symbol
        net = mx.sym.var('data')
        net = mx.gluon.model_zoo.vision.get_model(name="resnet152_v2", pretrained=False, classes=10)(net)
        net = mx.sym.SoftmaxOutput(net, name='softmax')
        self.network = net
        self.module = mx.mod.Module(symbol=net,
                            context=context,
                            data_names=['data'],
                            label_names=['softmax_label'])

    def train(self):

        mod = self.module
        train_iter, test_iter = CIFAR10(BATCH_SIZE, (3, 32, 32)).return_iters()

        mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
        mod.init_params(initializer=mx.init.Xavier(rnd_type='uniform', factor_type='out', magnitude=2))
        # mod.init_params(initializer=mx.init.Xavier())

        # defining regularization in the optimizer with wd
        opt = mx.optimizer.SGD(learning_rate=LEARNING_RATE, rescale_grad=(1.0/BATCH_SIZE), momentum=0.9, wd=0.0005)
        mod.init_optimizer(kvstore=KVSTORE, optimizer=opt)
        metric = mx.metric.create('acc')
        test_metric = mx.metric.create('acc')

        # key is epoch and value is learning rate
        lr_schedule = {0: 1*LEARNING_RATE, 82: 0.1*LEARNING_RATE, 123: 0.01*LEARNING_RATE, 300: 0.002*LEARNING_RATE}
        # lr_decay = 0.8
        # lr_schedule = {e: (LEARNING_RATE) * lr_decay ** i for i, e in enumerate(range(0, 300, 25))}
        print("lr_schedule: " + str(lr_schedule))

        for epoch in range(EPOCHS):
            epoch_tick = time.time()

            # update learning rate
            if epoch in lr_schedule.keys():
                mod._optimizer.lr = lr_schedule[epoch]
                print("Epoch %d, Changed learning rate to %s" % (epoch, str(lr_schedule[epoch])))

            train_iter.reset()
            metric.reset()
            for batch_idx, batch in enumerate(train_iter):
                batch_tick = time.time()
                mod.forward(batch, is_train=True)       # compute predictions
                mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
                mod.backward()                          # compute gradients
                mod.update() # update parameters
                # avoids logging on the first batch, hence batch_idx + 1
                if ((batch_idx + 1) % LOG_FREQUENCY) == 0:
                    pass
                    print('Epoch %d, Batch %d, Speed=%f' % (epoch, batch_idx, BATCH_SIZE/(time.time() - batch_tick)))

            print('Epoch %d, Duration=%f' % (epoch, time.time() - epoch_tick))
            # once per epoch, check training accuracy
            _, metric_value = metric.get()
            print('Epoch %d, Training accuracy=%f' % (epoch, metric_value))

            # once per epoch, check validation accuracy
            res = mod.score(test_iter, test_metric)
            _, metric_value = res[0]
            print('Epoch %d, Validation accuracy=%f' % (epoch, metric_value))

            if metric_value > EARLY_STOPPING_TEST_ACCURACY:
                print("Epoch %d, Reached early stopping target, stopping training." % (epoch))
                break


def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 image classification.')
    parser.add_argument('--gpus', type=int, default=0, help='Number of GPUs to use for training.')
    parser.add_argument('--total-batch-size', type=int, default=128, help='Batch size to use in total, spread equally across all GPUs.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate (for batch size of 128). Will be scaled linearly with batch size.')
    parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs to run, unless reach early stopping criteria.')
    parser.add_argument('--early-stopping-acc', type=float, default=0.85, help='Accuracy required on test set, to terminate training.')
    parser.add_argument('--seed', type=int, default=42, help='Seed to use for randomization.')
    parser.add_argument('--kvstore', type=str, default='device', help='KVStore to use for module.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    mx.random.seed(args.seed)

    EPOCHS = args.epochs
    GPUS = args.gpus
    BATCH_SIZE = args.total_batch_size # * max(1, GPUS) # total, not per GPU
    LEARNING_RATE = (args.lr / 128) * BATCH_SIZE  # examples found were using batch size 128 with learning rate 0.1 # lowered
    EARLY_STOPPING_TEST_ACCURACY = args.early_stopping_acc
    LOG_FREQUENCY = int(25600 / BATCH_SIZE)  # number of batches between each speed log
    KVSTORE = args.kvstore

    context = [mx.gpu(i) for i in range(GPUS)] if GPUS > 0 else [mx.cpu()]

    model = ResnetModel()
    model.train()