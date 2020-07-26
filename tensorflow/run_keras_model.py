import os

os.environ["HIP_VISIBLE_DEVICES"]="0"
# os.environ["TF_ROCM_RETURN_BEST_ALGO_ONLY"]="1"
# os.environ["TF_ROCM_USE_IMMEDIATE_MODE"]="1"
# os.environ["TF_CPP_MIN_VLOG_LEVEL"]="3"


import tensorflow as tf
import tensorflow.keras as keras

import time
import argparse


def set_keras_options(args):
    keras.backend.set_image_data_format('channels_first' if args.use_NCHW else 'channels_last')
    if args.use_keras_mixed_precision:
        keras.mixed_precision.experimental.set_policy('mixed_float16')
    

def get_model(args):

    input_shape = [args.num_batches, args.batch_size, 224, 224, 3]
    if args.use_NCHW:
        input_shape = [agrs.num_batches, args.batch_size, 3, 224, 224]

    if args.model == "resnet50":
        model = keras.applications.resnet.ResNet50()
    elif args.model == "resnet101":
        model = keras.applications.resnet.ResNet101()
    elif args.model == "resnet152":
        model = keras.applications.resnet.ResNet152()
    elif args.model == "densenet121":
        model = keras.applications.densenet.DenseNet121()
    elif args.model == "densenet169":
        model = keras.applications.densenet.DenseNet169()
    elif args.model == "densenet201":
        model = keras.applications.densenet.DenseNet201()
    else :
        print("Invalid model specified : ", args.model)
        model = None
        
    # model.summary()

    return model, input_shape


def run_model(args, model, input_shape):

    def run_once(local_model, local_x):
        start = time.time()
        preds = local_model.predict(local_x)
        stop = time.time()
        runtime = stop - start
        print (runtime)
        return preds, runtime
        
    
    dtype = tf.float32
    
    x = tf.random.uniform(input_shape, dtype=dtype)

    # first dummy run to get the auto-tuner run out of the way
    print("-"*50)
    print("model : ", args.model)
    print("-"*50)
    run_once(model, x[0])
    print("-"*50)

    num_runs = args.num_epochs * args.num_batches
    
    runtimes=[]
    for _ in range(args.num_epochs):
        for i in range(args.num_batches):
            preds, runtime = run_once(model, x[i])
            runtimes.append(runtime)

    runtimes.sort()
    print("-"*50)
    mean_batch_time = sum(runtimes)/num_runs
    print("Mean   = ", mean_batch_time)
    print("Median = ", runtimes[num_runs//2])
    print("-"*50)
    print("Images/sec = ", args.batch_size / mean_batch_time)
    print("-"*50)

    
def main(args):
    
    set_keras_options(args)
    
    model, input_shape = get_model(args)
    
    run_model(args, model, input_shape)
    

if __name__ == "__main__" :
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--use_keras_mixed_precision", action='store_true')
    parser.add_argument("--use_NCHW", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model", default="resnet50")
    args = parser.parse_args()

    main(args)
