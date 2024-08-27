import numpy as np 
from tensorflow import keras 
import tensorflow as tf

import tensorflow_privacy as tf_privacy

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

def train_shadow_models(x, y, model_creator, splits, out_path = './', BUFFER_SIZE = 1024, BATCH_SIZE = 256, INPUT_SHAPE = (32, 32, 3), AUTO = tf.data.AUTOTUNE, mae_callbacks = None, dstr_callbacks = None, dstr_optim = 'adam'):
    #for each index, 
    for i in range(len(splits)):

        (x_train, y_train), (x_untrain, y_untrain) = get_split(x, y, splits, i)
        print(f"Training samples: {len(x_train)}")
        print(f"Testing samples: {len(x_valid)}")

        model, model_dstr = model_creator()

        #TODO: make seperate datasets for MAE training and downstream classification
        train_ds_mae = tf.data.Dataset.from_tensor_slices(x_train)
        train_ds_mae = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

        valid_ds_mae = tf.data.Dataset.from_tensor_slices(x_untrain)
        valid_ds_mae = valid_ds.batch(BATCH_SIZE).prefetch(AUTO)

        #compile MAE model 
        opt_mae = keras.optimizer.Adam(amsgrad=True)
        
        model.compile(
            optimizer=opt, loss=keras.losses.MeanSquaredError(), metrics=["mae"]
        )

        #train mae model
        history = model.fit(
            train_ds_mae, epochs=EPOCHS, validation_data=valid_ds_mae, callbacks=mae_callbacks,
        )


        loss, mae = model.evaluate(valid_ds)
        print(f"Loss: {loss:.2f}")
        print(f"MAE: {mae:.2f}")

        #make dataset for downstream classification
        train_ds_dstr = tf.data.Dataset.from_tensor_slices(x_train, y_train)
        train_ds_dstr = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

        valid_ds_dstr = tf.data.Dataset.from_tensor_slices(x_untrain, y_untrain)
        valid_ds_dstr = valid_ds.batch(BATCH_SIZE).prefetch(AUTO)


        #train downstream model 
        if optim == 'adam':
            opt = keras.optimizers.Adam(amsgrad=True)
        elif optim == 'dpadam':
            opt = tf_privacy.DPKerasAdamOptimizer(
                l2_norm_clip=1.0, noise_multiplier = 0.5, num_microbatches=1, amsgrad=True
            )

        
        


        #infer logits and predictions
        #save the logits, predictions, membership ground truth
        #optionally save the model
        #clear model from memory 
