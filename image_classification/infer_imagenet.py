import argparse, time, logging, os
import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                    help='training and validation pictures to use.')
parser.add_argument('--rec-val', type=str, default='/media/ramdisk/data/val-passthrough.rec',
                    help='the validation data')
parser.add_argument('--rec-val-idx', type=str, default='/media/ramdisk/data/val-passthrough.idx',
                    help='the index of validation data')
parser.add_argument('--use-rec', action='store_true',
                    help='use image record iter for data input. default is false.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--dataset-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')
parser.add_argument('--num_gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--num-epochs', type=int, default=3,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--lr-mode', type=str, default='step',
                    help='learning rate scheduler mode. options are step, poly.')
parser.add_argument('--lr-poly-power', type=int, default=2,
                    help='if learning rate scheduler mode is poly, then power is used')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epoches at which learning rate decays. default is 40,60.')
parser.add_argument('--warmup-lr', type=float, default=0.0,
                    help='starting warmup learning rate. default is 0.0.')
parser.add_argument('--warmup-epochs', type=int, default=0,
                    help='number of warmup epochs.')
parser.add_argument('--last-gamma', action='store_true',
                    help='whether to initialize the gamma of the last BN layer in each bottleneck to zero')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--use_se', action='store_true',
                    help='use SE layers or not in resnext. default is false.')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--log-interval', type=int, default=50,
                    help='Number of batches to wait before logging.')
parser.add_argument('--save-frequency', type=int, default=0,
                    help='frequency of model saving.')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--logging-dir', type=str, default='logs',
                    help='directory of training logs')
parser.add_argument('--kvstore', type=str, default='nccl')
parser.add_argument('--top-k', type=int, default=0, help='give 5 for top5 accuracy, if 0 only prints top1 accuracy')
opt = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info(opt)

batch_size = opt.batch_size
classes = 1000

num_gpus = opt.num_gpus
context = [mx.cpu()]
num_workers = opt.num_workers

kv = mx.kv.create(opt.kvstore)
model_name = opt.model

kwargs = {'ctx': context, 'pretrained': opt.use_pretrained, 'classes': classes}
net = get_model(model_name, **kwargs)

def get_data_rec(rec_val, rec_val_idx, batch_size, num_workers):
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    mean_rgb = [123.68, 116.779, 103.939]
    
    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label
    
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        path_imgidx         = rec_val_idx,
        preprocess_threads  = num_workers,
        shuffle             = False,
        batch_size          = batch_size,
        resize              = 256,
        label_width         = 1,
        rand_crop           = False,
        rand_mirror         = False,
        data_shape          = (3, 224, 224),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2]
    )
    return val_data, batch_fn

def get_data_loader(data_dir, batch_size, num_workers):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        return data, label
    if opt.mode == 'symbolic':
        val_data = mx.io.NDArrayIter(
            mx.nd.random.normal(shape=(opt.dataset_size, 3, 224, 224)),
            label=mx.nd.array(range(opt.dataset_size)),
            batch_size=batch_size,
        )
        transform_test = transforms.Compose([
            transforms.Resize(256, keep_ratio=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        val_data = gluon.data.DataLoader(
            imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return val_data, batch_fn


if opt.use_rec:
    val_data, batch_fn = get_data_rec(opt.rec_val, opt.rec_val_idx, batch_size, num_workers)
else:
    val_data, batch_fn = get_data_loader(opt.data_dir, batch_size, num_workers)

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

def infer(ctx):
    for i, batch in enumerate(val_data):
        btic = time.time()
        data, label = batch_fn(batch, ctx)
        outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
        acc_top1.update(label, outputs)
        #acc_top5.update(label, outputs)
        logging.info('Batch [%d]'%(i))
        logging.info('Top 1 accuracy: %d'%(acc_top1.get()[1]))
        #logging.info('Top 5 accuracy: %d'%(acc_top5.get()[1]))
        time_taken = time.time() - btic
        if i<20:
            logging.info('warmup_throughput: %d samples/sec warmup_time %f'%(
                         int(batch_size / time_taken), time_taken))
        else:
            logging.info('Speed: %d samples/sec Time cost=%f'%(
                         int(batch_size / time_taken), time_taken))
    return

def main():
    if opt.mode == 'symbolic':
        data = mx.sym.var('data')
        if opt.dtype == 'float16':
            data = mx.sym.Cast(data=data, dtype=np.float16)
            net.cast(np.float16)
        out = net(data)
        if opt.dtype == 'float16':
            out = mx.sym.Cast(data=out, dtype=np.float32)
        softmax = mx.sym.SoftmaxOutput(out, name='softmax')
        mod = mx.mod.Module(softmax, context=context)
        net.hybridize()
        net(mx.nd.random_normal(shape=(1,3,256,256)))
        net.export('preresnet50',0)
        sym, arg_params, aux_params = mx.model.load_checkpoint('preresnet50',0)
        mod.bind(data_shapes=val_data.provide_data, label_shapes=val_data.provide_label)
        mod.set_params(arg_params, aux_params)
        mod.score(
            eval_data=val_data,
            eval_metric=mx.metric.Accuracy(),
            batch_end_callback=mx.callback.Speedometer(batch_size, opt.log_interval)
        )
    else:
        if opt.mode == 'hybrid':
            net.hybridize(static_alloc=True, static_shape=True)
        infer(context)


if __name__ == '__main__':
    main()
