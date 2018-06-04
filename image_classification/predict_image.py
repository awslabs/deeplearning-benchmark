
# # Predict with pre-trained models

import argparse, time
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import mxnet as mx
import cv2
from collections import namedtuple


class InferenceTesting(object):
    def __init__(self, opt):
        self.Batch = namedtuple('Batch',['data'])
        self.model = opt.model
        self.iterations = opt.iterations
        self.model_path = opt.model_path
        self.url = opt.url

    def downloadModel(self):
        network = self.model.split('-')[0]
        layers = self.model.split('-')[1]
        base_path = "{}{}/{}-layers/".format(self.model_path,network, layers)
        print(base_path)
        model_json_path = "{}{}-symbol.json".format(base_path, self.model)
        model_params_path = "{}{}-0000.params".format(base_path, self.model)
        model_synset_path = "{}{}/synset.txt".format(self.model_path,network)
        print(model_synset_path)
        print(model_json_path)
        print(model_params_path)

        try:
            mx.test_utils.download(model_json_path)
            mx.test_utils.download(model_params_path)
            mx.test_utils.download(model_synset_path)
        except Exception as e:
            print("Error in downloading the models {}".format(e))

    def __loadModel(self):
        sym, arg_params, aux_params = mx.model.load_checkpoint(self.model, 0)
        cont = mx.gpu() if opt.use_gpus > 0 else mx.cpu()
        self.mod = mx.mod.Module(symbol=sym, context=cont, label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))],
                 label_shapes=self.mod._label_shapes)
        self.mod.set_params(arg_params, aux_params, allow_missing=True)
        with open('synset.txt', 'r') as f:
            self.labels = [l.rstrip() for l in f]

    def __getImage(self):
        try:
            fname = mx.test_utils.download(self.url)
        except Exception as e:
            print("Error in downloading the image {}".format(e))
        img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        if img is None:
            return None
        # convert into format (batch, RGB, width, height)
        img = cv2.resize(img, (224, 224))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]
        return img

    def predict(self):
        self.__loadModel()
        img = self.__getImage()
        if img is None:
            print("Error: Can not load the image")
        # compute the predict probabilities
        for i in range(self.iterations):
            tic = time.time()
            self.mod.forward(self.Batch([mx.nd.array(img)]))
            pred_time_ms = (time.time() - tic) * 1000
            print ("Prediction-Time: {} milliseconds".format(pred_time_ms))

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--model', type=str, default='resnet-50',
                        help='pre-trained model')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of times inference is generated for a single image.')
    parser.add_argument('--model_path', type=str, default='http://data.mxnet.io/models/imagenet/',
                        help='Default path to load the model')
    parser.add_argument('--url', type=str, default='http://writm.com/wp-content/uploads/2016/08/Cat-hd-wallpapers.jpg',
                        help='Url to the image file')
    parser.add_argument('--use_gpus', type=int, default=0,
                        help="Indicate whether to use gpus")
    opt = parser.parse_args()

    print(opt)

    # Following sleep is added so that process runs until cpu-gpu profiler process starts.
    infer = InferenceTesting(opt)
    infer.downloadModel()
    infer.predict()
    time.sleep(10)
    print ("Done")
    exit()

