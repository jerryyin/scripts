import os

os.environ["HIP_VISIBLE_DEVICES"]="0"
# os.environ["TF_ROCM_RETURN_BEST_ALGO_ONLY"]="1"
# os.environ["TF_ROCM_USE_IMMEDIATE_MODE"]="1"
#os.environ["TF_CPP_MIN_VLOG_LEVEL"]="3"


import tensorflow as tf
import tensorflow.keras as keras

import time
import argparse


def run_mnist_model(args):
    
    num_units = 4096
        
    if args.use_keras_mixed_precision:
        keras.mixed_precision.experimental.set_policy('mixed_float16')
        
    inputs = keras.Input(shape=(784,), name='digits')
    
    dense_1 = keras.layers.Dense(num_units, activation='relu', name='dense_1')
    print("-"*50)
    print("dense_1 compute type  = ", dense_1._compute_dtype)
    print("dense_1 variable type = ", dense_1._dtype)
    x = dense_1(inputs)
    
    dense_2 = keras.layers.Dense(num_units, activation='relu', name='dense_2')
    print("-"*50)
    print("dense_2 compute type  = ", dense_2._compute_dtype)
    print("dense_2 variable type = ", dense_2._dtype)
    x = dense_2(x)
    
    dense_logits = keras.layers.Dense(10, name='dense_logits')
    print("-"*50)
    print("dense_logits compute type  = ", dense_logits._compute_dtype)
    print("dense_logits variable type = ", dense_logits._dtype)
    x = dense_logits(x)
    
    # explicitly set the compute dtype to float32, that overrides the global policy default
    predictions = keras.layers.Activation('softmax', dtype='float32', name='predictions')
    print("-"*50)
    print("predictions compute type  = ", predictions._compute_dtype)
    print("predictions variable type = ", predictions._dtype)
    
    outputs = predictions(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    loss = 'sparse_categorical_crossentropy'
    
    optimizer = keras.optimizers.RMSprop()
    if args.use_graph_rewrite_mixed_precision:
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
        
    metrics = ['accuracy']
        
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    dtype = tf.float32
    
    history = model.fit(x_train, y_train, batch_size=8192, epochs=20, validation_split=0.2)
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])


    
def main(args):

    run_mnist_model(args)
    

if __name__ == "__main__" :
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--use_keras_mixed_precision", action='store_true')
    parser.add_argument("-g", "--use_graph_rewrite_mixed_precision", action='store_true')
    args = parser.parse_args()

    if (args.use_keras_mixed_precision and args.use_graph_rewrite_mixed_precision):
        print("Please specify enabling AMP via only one of either Keras or Graph Rewrite")
        
    main(args)
