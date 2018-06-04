import os
import subprocess
import glob
import time
import numpy as np

def get_model_input(model_dir):
    import onnx
    from onnx import numpy_helper

    model_inputs = []
    for test_data_npz in glob.glob(
            os.path.join(model_dir, 'test_data_*.npz')):
        test_data = np.load(test_data_npz, encoding='bytes')
        model_inputs = list(test_data['inputs'])

    for test_data_dir in glob.glob(
            os.path.join(model_dir, "test_data_set*")):
        inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
        for i in range(inputs_num):
            input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
            tensor = onnx.TensorProto()
            with open(input_file, 'rb') as f:
                tensor.ParseFromString(f.read())
            model_inputs.append(numpy_helper.to_array(tensor))

    return model_inputs

def profile_model(model_path, test_data, context):   
    import mxnet as mx

    sym, arg_params, aux_params = mx.contrib.onnx.import_model(model_path)
    ctx = mx.gpu() if context == "gpu" else mx.cpu()
    data_names = [graph_input for graph_input in sym.list_inputs()
                  if graph_input not in arg_params and graph_input not in aux_params]

    inference_time = 0
    for data_idx, _ in enumerate(test_data):
        data_shapes = []
        for idx, input_name in enumerate(data_names):
            data_shapes.append((input_name, test_data[idx+data_idx].shape))

        # create a module
        mod = mx.mod.Module(symbol=sym, data_names=data_names, context=ctx, label_names=None)
        mod.bind(for_training=False, data_shapes=data_shapes, label_shapes=None)

        # initializing parameters for calculating result of each individual node
        if arg_params is None and aux_params is None:
            mod.init_params()
        else:
            mod.set_params(arg_params=arg_params, aux_params=aux_params)

        data_forward = []
        for idx, input_name in enumerate(data_names):
            val = test_data[idx+data_idx]
            data_forward.append(mx.nd.array(val))

        start = time.time()
        mod.forward(mx.io.DataBatch(data_forward))
        total_time =  (time.time() - start)*1000
        #total_time = "{:.9f}".format(total_time)
        # log_data.update({test_data_name: total_time})
        inference_time += total_time
    
    avg_inference_time = inference_time/len(test_data)
    return avg_inference_time

if __name__ == '__main__':
       from sys import argv
       ctx = str(argv[1])
       for directory in os.listdir("./models"):
           model_dir = os.path.join("./models", directory)
           if os.path.isdir(model_dir):
               model_path = os.path.join(model_dir, "model.onnx")
               test_data = get_model_input(model_dir)

               profile_data = profile_model(model_path, test_data, ctx)

               avg_inference_time = profile_model(model_path, test_data, ctx)

               print('Average_inference_time_{}_{}: {:.9f}'.format(directory, ctx, avg_inference_time))

