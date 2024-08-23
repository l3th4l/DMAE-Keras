import numpy as np 
from tensorflow import keras 
import tensorflow as tf

def shadow_split_indices(total_size, num_splits = 10, split_size = None, split_p = 0.1):
    if not split_size:
        split_size = int(total_size * split_p)

    splits = []

    for i in range(num_splits):
        split = np.random.choice(total_size, size = split_size*2)
        splits.append((split[:num_splits], split[num_splits:]))

def get_split(x, y, splits, id):
    split_train, split_untrain = splits[id]

    x_train = x[split_train]
    x_untrain = x[split_untrain]

    y_train = y[split_train]
    y_untrain = y[split_untrain]

    return (x_train, y_train), (x_untrain, y_untrain)

def train_shadow_models(x, y, model_creator, splits, out_path = './', BUFFER_SIZE = 1024, BATCH_SIZE = 256, INPUT_SHAPE = (32, 32, 3), AUTO = tf.data.AUTOTUNE):
    #for each index, 
    for i in range(len(splits)):

        (x_train, y_train), (x_untrain, y_untrain) = get_split(x, y, splits, i)
        print(f"Training samples: {len(x_train)}")
        print(f"Testing samples: {len(x_valid)}")

        #TODO: make seperate datasets for MAE training and downstream classification
        train_ds = tf.data.Dataset.from_tensor_slices(x_train)
        train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

        valid_ds = tf.data.Dataset.from_tensor_slices(x_valid)
        valid_ds = valid_ds.batch(BATCH_SIZE).prefetch(AUTO)

        model = model_creator()

        #train a shadow model, 
        #infer logits and predictions
        #save the logits, predictions, membership ground truth
        #optionally save the model
        #clear model from memory 
