import argparse
import mxnet as mx
from mxnet import ndarray as nd
import time
import math

input_height = 512
input_width = 512
channels = 3


def get_argument_parser():
    parser = argparse.ArgumentParser("Parameters for running SSD")
    parser.add_argument("--context",
                        help="The context to run on ", default="cpu")
    parser.add_argument("--modelPathPrefix",
                        help="The model path prefix", default="/tmp/resnet50_ssd/resnet50_ssd_model")
    parser.add_argument("--inputImagePath",
                        help="Path of the input image to be tested", default="/tmp/resnet50_ssd/images/dog.jpg")
    parser.add_argument("--batchSize",
                        help="batch size of the images to be tested", default=4, type=int)
    parser.add_argument("--times",
                        help="Number of times to run the benchmark", default=10, type=int)
    return parser.parse_args()


def load_model(model_path_prefix, ctx, batchSize, epoch_num = 0):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_path_prefix, epoch_num)
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    data_shape = [('data', (batchSize, channels, input_height, input_width))]
    label_shape = mod._label_shapes

    mod.bind(for_training=False, data_shapes=data_shape, label_shapes=label_shape)
    mod.set_params(arg_params, aux_params)
    return mod


def get_single_image_ndarray(input_image_path, ctx):
    img = mx.image.imread(input_image_path)
    img = mx.image.imresize(img, input_height, input_width)
    img = img.transpose((2, 0, 1))  # Channel first
    img = img.expand_dims(axis=0)  # Add a new axis

    return img.as_in_context(ctx)


def get_batch_image_ndarray(input_image_path, ctx, batchSize):
    img = mx.image.imread(input_image_path)
    img = mx.image.imresize(img, input_height, input_width)
    img = img.transpose((2, 0, 1))  # Channel first
    img = img.expand_dims(axis=0)  # Add a new axis

    result_img = img
    for i in range(1, batchSize):
        result_img = nd.concat(result_img, img, dim=0)

    return result_img.as_in_context(ctx)


def run_single_inference(model_path_prefix, input_image_path, ctx, times):
    model = load_model(model_path_prefix, ctx, batchSize=1)
    data = get_single_image_ndarray(input_image_path, ctx)

    print(data.shape)
    data_iter = mx.io.NDArrayIter([data], None, 1)

    print ("warming up the system")
    for i in range(1, 5):
        prediction_op = model.predict(data_iter)
        prediction_op.wait_to_read()

    print ("Warm up done")

    time_readings = list()

    for i in range(1, times):
        start = time.time()
        prediction_op = model.predict(data_iter)
        prediction_op.wait_to_read()
        # results = prediction_op[0].asnumpy()
        # print results[0]
        end = time.time()
        time_readings.append(end - start)
        print ("Inference time at iteration %d is %f ms \n" % (i, (end - start) * 1000))

        time_readings.sort()
    return time_readings


def run_batch_inference(model_path_prefix, input_image_path, ctx, times, batchSize):
    model = load_model(model_path_prefix, ctx, batchSize=batchSize)
    data = get_batch_image_ndarray(input_image_path, ctx, batchSize)

    print (data.shape)
    data_iter = mx.io.NDArrayIter([data], None, batchSize)

    print ("warming up the system")
    for i in range(1, 5):
        prediction_op = model.predict(data_iter)
        prediction_op.wait_to_read()

    print ("Warm up done")

    time_readings = list()

    for i in range(1, times):
        start = time.time()
        prediction_op = model.predict(data_iter)
        prediction_op.wait_to_read()
        # results = prediction_op.asnumpy()
        # print (results[0])
        end = time.time()
        time_readings.append(end - start)
        print ("Inference time at iteration %d is %f ms \n" % (i, (end - start) * 1000))

        time_readings.sort()
    return time_readings


def percentile(val, arr):
    idx = int(math.ceil((len(arr) - 1) * val / 100.0))
    return arr[idx]


def emit_metrics(arr, metrics_prefix):

    p50 = percentile(50, arr) * 1000
    p90 = percentile(90, arr) * 1000
    p99 = percentile(99, arr) * 1000
    average = sum(arr) / len(arr) * 1000
    # converting to msec by multiplying

    print("\n%s_p99 %1.2f, %s_p90 %1.2f, %s_p50 %1.2f, %s_average %1.2f\n" %
          (metrics_prefix, p99, metrics_prefix, p90, metrics_prefix, p50, metrics_prefix, average))


if __name__ == "__main__":
    args = get_argument_parser()

    ctx = mx.cpu(0) if args.context == "cpu" else mx.gpu(0)

    print("Running single inference")
    single_inference_times = run_single_inference(args.modelPathPrefix, args.inputImagePath, ctx,
                                                  args.times)
    emit_metrics(single_inference_times, "single_inference")

    print("Running batch inference with batch size : %d" % args.batchSize)
    batch_inference_1x_times = run_batch_inference(args.modelPathPrefix, args.inputImagePath, ctx,
                                                   args.times, args.batchSize)
    emit_metrics(batch_inference_1x_times, "batch_inference_1x")

    print("Running batch inference with batch size : %d" % (2 * args.batchSize))
    batch_inference_2x_times = run_batch_inference(args.modelPathPrefix, args.inputImagePath, ctx,
                                                   args.times, 2 * args.batchSize)
    emit_metrics(batch_inference_2x_times, "batch_inference_2x")

    print("Running batch inference with batch size : %d" % (4 * args.batchSize))
    batch_inference_4x_times = run_batch_inference(args.modelPathPrefix, args.inputImagePath, ctx,
                                                   args.times, 4 * args.batchSize)
    emit_metrics(batch_inference_4x_times, "batch_inference_4x")
