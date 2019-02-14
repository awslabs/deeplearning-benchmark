import argparse, time
import numpy as np
import mxnet as mx

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

class InferenceHelper(object):
    def __init__(self, opt, batch_size):
        # Pre-processing details
        self.input_shape = (224, 224)
        self.norm_mean = (0.485, 0.456, 0.406)
        self.norm_std = (0.229, 0.224, 0.225)

        self.model_path = opt.model_path
        self.model_name = opt.model_name
        self.iterations = opt.iterations
        self.is_end_to_end = opt.end_to_end.lower() == 'true'
        self.batch_size = batch_size

        self.ctx = mx.gpu(0) if int(opt.use_gpus) > 0 else mx.cpu()

        # If end to end model, pre-processing is part of the network,
        # we take raw input data in channels last format.
        if self.is_end_to_end:
            self.input_data_shape = (self.batch_size, 224, 224, 3)
        else:
            self.input_data_shape = (self.batch_size, 3, 224, 224)
        
        print("Running end-to-end v/s non-end-to-end inference benchmarks with below configuration: ")
        print("Model Path - {}".format(self.model_path))
        print("Model Name - {}".format(self.model_name))
        print("Context - {}".format(self.ctx))
        print("Is end to end model - {}".format(self.is_end_to_end))
        print("Input shape - {}".format(self.input_data_shape))
        print("Inference batch size - {}".format(self.batch_size))
        print("Iterations - {}".format(self.iterations))

    def download_model(self):
        model_json_path = "{}/{}-symbol.json".format(self.model_path, self.model_name)
        model_params_path = "{}/{}-0000.params".format(self.model_path, self.model_name)
        print("Downloading the following model files...")
        print(model_json_path)
        print(model_params_path)

        try:
            mx.test_utils.download(model_json_path)
            mx.test_utils.download(model_params_path)
        except Exception as e:
            print("ERROR: Failed to download the models. {}".format(e))
    
    def load_model(self):
        sym, arg_params, aux_params = mx.model.load_checkpoint(self.model_name, 0)
        self.mod = mx.mod.Module(symbol=sym, context=self.ctx, label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', self.input_data_shape)],
                      label_shapes=self.mod._label_shapes)
        self.mod.set_params(arg_params, aux_params, allow_missing=True)

    def __preprocess_data(self, raw_data):
        """Data pre-processing. Process 1 sample at a time.
        Used in non end to end model benchmarking.
        """
        # Resize -> ToTensor -> Normalize
        processed_data = [] #nd.array(shape = data.shape[0])
        for i in range(raw_data.shape[0]):
            img = mx.image.imresize(raw_data[i], self.input_shape[0], self.input_shape[1])
            img = img.astype(np.float32)
            img /= 255
            img = mx.image.color_normalize(img,
                                           mean=mx.nd.array(self.norm_mean),
                                           std=mx.nd.array(self.norm_std))
            img = mx.nd.transpose(img, (2, 0, 1))
            processed_data.append(img)
        return mx.nd.stack(*processed_data)

    def inference_benchmark(self):
        total_time_ms = 0.0
        for i in range(self.iterations):
            # Generate a synthetic data
            data = mx.nd.random.uniform(0, 255, (self.batch_size, 300, 300, 3)).astype(dtype=np.uint8)
            tic = time.time()
            # End to end models have pre-processing as part of the model.
            # Do pre-processing only for non end to end models. 
            if not self.is_end_to_end:
                data = self.__preprocess_data(data)
            data = data.as_in_context(self.ctx)
            self.mod.forward(Batch([data]))
            self.mod.get_outputs()[0].wait_to_read()
            pred_time_ms = (time.time() - tic) * 1000
            total_time_ms += pred_time_ms

        # Return Average prediction time per sample
        return total_time_ms/(self.batch_size * self.iterations)

if  __name__ == '__main__':
    # Model Path - https://s3.us-east-2.amazonaws.com/mxnet-public/end_to_end_models/
    # Model Names:
    # End to end model name - resnet18_v1_end_to_end
    # Non end to end model name - resnet18_v1
    parser = argparse.ArgumentParser(description='Inference speed test for image classification.')
    parser.add_argument('--model_path', type=str,
                        help='Path to download the model')
    parser.add_argument('--model_name', type=str, default='resnet18_v1',
                        help='Name of the model. This will be used to download the right symbol and params files from model_path')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of times inference is generated for a single image.')
    parser.add_argument('--use_gpus', type=int, default=0,
                        help="Indicate whether to use gpus")
    parser.add_argument('--end_to_end', type=str, default='false', help="If the model is end to end or not.")
    opt = parser.parse_args()
    
    # Following sleep is added so that process runs until cpu-gpu profiler process starts.
    time.sleep(10)

    # Run benchmarks for single request inference
    inference_helper = InferenceHelper(opt, batch_size=1)
    inference_helper.download_model()
    inference_helper.load_model()
    avg_pred_time_per_sample = inference_helper.inference_benchmark()
    print ("Single Inference: Average prediction time per sample: {} ms".format(avg_pred_time_per_sample))

    # Run benchmarks for batch inference
    inference_helper = InferenceHelper(opt, batch_size=25)
    inference_helper.download_model()
    inference_helper.load_model()
    avg_pred_time_per_sample = inference_helper.inference_benchmark()
    print ("Batch Inference: Average prediction time per sample: {} ms".format(avg_pred_time_per_sample))

    print ("Done")
    exit()
