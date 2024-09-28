import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
#from tensorflow.keras import ops

import numpy as np
import matplotlib.pyplot as plt
import random
from os import name


class Patches(L.Layer):
    def __init__(self, patch_size=PATCH_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.resize = L.Reshape((-1, patch_size * patch_size * 3))

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = self.resize(patches)
        return patches

    def show_patched_image(self, images, patches):
        idx = np.random.choice(patches.shape[0])
        print(f"Index selected: {idx}.")

        plt.figure(figsize=(4, 4))
        plt.imshow(keras.utils.array_to_img(images[idx]))
        plt.suptitle('Original Image')
        plt.axis("off")
        plt.show()

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        plt.suptitle('Patches')
        for i, patch in enumerate(patches[idx]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = tf.reshape(patch, (self.patch_size, self.patch_size, 3))
            plt.imshow(keras.utils.img_to_array(patch_img))
            plt.axis("off")
        plt.show()

        return idx

    def reconstruct_from_patch(self, patch):
        num_patches = patch.shape[0]
        n = int(np.sqrt(num_patches))
        patch = tf.reshape(patch, (num_patches, self.patch_size, self.patch_size, 3))
        rows = tf.split(patch, n, axis=0)
        rows = [tf.concat(tf.unstack(x), axis=1) for x in rows]
        reconstructed = tf.concat(rows, axis=0)
        return reconstructed


class PatchEncoder(L.Layer):
    def __init__(
        self,
        patch_size=PATCH_SIZE,
        projection_dim=ENC_PROJECTION_DIM,
        mask_proportion=MASK_PROPORTION,
        downstream=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion
        self.downstream = downstream

        self.mask_token = tf.Variable(
            tf.random.normal([1, patch_size * patch_size * 3]), trainable=True
        )

    def build(self, input_shape):
        (_, self.num_patches, self.patch_area) = input_shape
        self.projection = L.Dense(units=self.projection_dim)
        self.position_embedding = L.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim
        )
        self.num_mask = int(self.mask_proportion * self.num_patches)

    def call(self, patches):
        batch_size = tf.shape(patches)[0]
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embeddings = self.position_embedding(positions[tf.newaxis, ...])
        pos_embeddings = tf.tile(
            pos_embeddings, [batch_size, 1, 1]
        )
        patch_embeddings = (
            self.projection(patches) + pos_embeddings
        )
        if self.downstream:
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            unmasked_embeddings = tf.gather(
                patch_embeddings, unmask_indices, axis=1, batch_dims=1
            )
            unmasked_positions = tf.gather(
                pos_embeddings, unmask_indices, axis=1, batch_dims=1
            )
            masked_positions = tf.gather(
                pos_embeddings, mask_indices, axis=1, batch_dims=1
            )
            mask_tokens = tf.repeat(self.mask_token, repeats=self.num_mask, axis=0)
            mask_tokens = tf.repeat(
                mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0
            )

            masked_embeddings = self.projection(mask_tokens) + masked_positions
            return (
                unmasked_embeddings,
                masked_embeddings,
                unmasked_positions,
                mask_indices,
                unmask_indices,
            )

    def get_random_indices(self, batch_size):
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_patches)), axis=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices

    def generate_masked_image(self, patches, unmask_indices):
        idx = np.random.choice(patches.shape[0])
        patch = patches[idx]
        unmask_index = unmask_indices[idx]
        new_patch = np.zeros_like(patch)
        count = 0
        for i in range(unmask_index.shape[0]):
            new_patch[unmask_index[i]] = patch[unmask_index[i]]
        return new_patch, idx


def dense_projection(x, dropout_rate, hidden_units, name = None):
    for units in hidden_units:
        if name:
            #print(name + '_' + str(units))
            x = L.Dense(units, activation=tf.nn.gelu, name = name + '_' + str(units))(x)
        else:
            x = L.Dense(units, activation=tf.nn.gelu)(x)
        x = L.Dropout(dropout_rate)(x)
    return x


#Teacher Block 
def create_vit_classifier():
    inputs = keras.Input(shape=INPUT_SHAPE)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(PATCH_SIZE)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(NUM_PATCHES, teacher_config['projection_dim'], downstream=True)(patches)

    # Create multiple layers of the Transformer block.
    for i in range(teacher_config['transformer_layers']):
        # Layer normalization 1.
        x1 = L.LayerNormalization(epsilon=1e-6, name = f"vit_lnorm_{i}__1")(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = L.MultiHeadAttention(
            num_heads=teacher_config['num_heads'], key_dim=teacher_config['projection_dim'], dropout=0.1,
            name=f"vit_attention_{i}"
        )(x1, x1)
        # Skip connection 1.
        x2 = L.Add(name = f"vit_add_{i}")([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = L.LayerNormalization(epsilon=1e-6, name = f"vit_lnorm_{i}__2")(x2)
        # MLP.
        x3 = dense_projection(x3, hidden_units=teacher_config['transformer_units'], dropout_rate=0.1, name = f"vit_dense_{i}")
        # Skip connection 2.
        encoded_patches = L.Add(name = f"vit_add_{i}_2")([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = L.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = L.Flatten()(representation)
    representation = L.Dropout(0.5)(representation)
    # Add MLP.
    features = dense_projection(representation, hidden_units=teacher_config['mlp_head_units'], dropout_rate=0.5)
    # Classify outputs.
    num_classes = 100
    logits = L.Dense(100)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    fe_model = keras.Model(inputs=inputs, outputs=model.get_layer("vit_dense_5_128").output, name="vit_interim")
    return model, fe_model

#MAE Encoder Block 
def create_encoder(num_heads=ENC_NUM_HEADS, num_layers=ENC_LAYERS, _teacher_config = teacher_config):
    inputs = L.Input((None, ENC_PROJECTION_DIM))
    x = inputs

    for i in range(num_layers):
        x1 = L.LayerNormalization(epsilon=LAYER_NORM_EPS, name = f"mae_lnorm1_{i}_enc")(x)
        attention_output = L.MultiHeadAttention(
            num_heads=num_heads, key_dim=ENC_PROJECTION_DIM, dropout=0.1,
            name=f"mae_attention_{i}_enc"
        )(x1, x1)
        x2 = L.Add(name = f"mae_add1_{i}_enc")([attention_output, x])
        x3 = L.LayerNormalization(epsilon=LAYER_NORM_EPS, name = f"mae_lnorm2_{i}_enc")(x2)
        x3 = dense_projection(x3, hidden_units=ENC_TRANSFORMER_UNITS, dropout_rate=0.1, name = f"mae_dense_{i}_enc")
        x = L.Add(name = f"mae_add2_{i}_enc")([x3, x2])

    outputs = L.LayerNormalization(epsilon=LAYER_NORM_EPS, name = f"mae_lnorm_out")(x)
    main_model = keras.Model(inputs, outputs, name="mae_encoder")

    #project intermediate output to teacher dimensions
    attention_output = main_model.get_layer("mae_attention_4_enc").output
    out_project = dense_projection(outputs, hidden_units=teacher_config['transformer_units'], dropout_rate=0.01, name = f"mae_dense_project")

    intermediate_layer_model = keras.Model(inputs=main_model.input, outputs=out_project, name="mae_encoder_interim")
    return main_model, intermediate_layer_model

def create_decoder(
    num_layers=DEC_LAYERS, num_heads=DEC_NUM_HEADS, image_size=IMAGE_SIZE
):
    inputs = L.Input((NUM_PATCHES, ENC_PROJECTION_DIM))
    x = L.Dense(DEC_PROJECTION_DIM)(inputs)

    for i in range(num_layers):
        x1 = L.LayerNormalization(epsilon=LAYER_NORM_EPS, name = f"mae_lnorm1_{i}_dec")(x)
        attention_output = L.MultiHeadAttention(
            num_heads=num_heads, key_dim=DEC_PROJECTION_DIM, dropout=0.1, name = f"mae_attention_{i}_dec"
        )(x1, x1)
        x2 = L.Add(name = f"mae_add1_{i}_dec")([attention_output, x])
        x3 = L.LayerNormalization(epsilon=LAYER_NORM_EPS, name = f"mae_lnorm2_{i}_dec")(x2)
        x3 = dense_projection(x3, hidden_units=DEC_TRANSFORMER_UNITS, dropout_rate=0.1, name = f"mae_dense_{i}_dec")
        x = L.Add(name = f"mae_add2_{i}_dec")([x3, x2])

    x = L.LayerNormalization(epsilon=LAYER_NORM_EPS, name = f"mae_lnorm_out_dec")(x)
    x = L.Flatten(name = f"mae_flatten_out_dec")(x)
    pre_final = L.Dense(units=image_size * image_size * 3, activation="sigmoid")(x)
    outputs = L.Reshape((image_size, image_size, 3))(pre_final)

    return keras.Model(inputs, outputs, name="mae_decoder")