import os
from datetime import datetime
import tensorflow as tf
from tf_siren import siren_mlp
import SimpleITK as sitk
import numpy as np

SAMPLING_RATIO = 0.3
BATCH_SIZE = 16384 #1048576 #8192
EPOCHS = 50000

# Image - CT scan
def normalize_array(arr):
    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return norm_arr


def build_train_tensors(sampled_pixel_count, rows, cols, depth):
    img_mask_x = tf.random.uniform([sampled_pixel_count], maxval=rows, seed=0, dtype=tf.int32)
    img_mask_y = tf.random.uniform([sampled_pixel_count], maxval=cols, seed=1, dtype=tf.int32)
    img_mask_z = tf.random.uniform([sampled_pixel_count], maxval=depth, seed=1, dtype=tf.int32)

    img_mask_x = tf.expand_dims(img_mask_x, axis=-1)
    img_mask_y = tf.expand_dims(img_mask_y, axis=-1)
    img_mask_z = tf.expand_dims(img_mask_z, axis=-1)

    img_mask_idx = tf.concat([img_mask_x, img_mask_y, img_mask_z], axis=-1)
    img_train = tf.gather_nd(img_ground_truth, img_mask_idx, batch_dims=0)

    img_mask_x = tf.cast(img_mask_x, tf.float32) / rows
    img_mask_y = tf.cast(img_mask_y, tf.float32) / cols
    img_mask_z = tf.cast(img_mask_z, tf.float32) / depth

    img_mask = tf.concat([img_mask_x, img_mask_y, img_mask_z], axis=-1)

    return img_mask, img_train


# Build model
def get_compiled_model(num_steps):
    model = siren_mlp.SIRENModel(units=256, final_units=channels, final_activation='sigmoid', num_layers=5, w0=1.0, w0_initial=30.0)

    # instantiate model
    _ = model(tf.zeros([1, 3]))

    model.summary()

    learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(5e-4, decay_steps=num_steps, end_learning_rate=5e-5, power=2.0)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)  # Sum of squared error
    model.compile(optimizer, loss=loss)
    return model


def make_or_restore_model(checkpoint_dir, num_steps):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return tf.keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model(num_steps)


def run_training(epochs, checkpoint_dir, num_steps):
    strategy = tf.distribute.MirroredStrategy()
    print("Number of available devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = make_or_restore_model(checkpoint_dir, num_steps)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/ckpt-model-{epoch}.keras", monitor='loss', save_freq="epoch", save_best_only=True), #(checkpoint_dir + 'model.weights.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min'),
        tf.keras.callbacks.TensorBoard(logdir, update_freq='batch', profile_batch=20)
    ]

    model.fit(train_dataset, epochs=epochs, callbacks=callbacks, verbose=2)


img_filepath = 'data/image_ct.mha'
img_raw = normalize_array(sitk.GetArrayFromImage(sitk.ReadImage(img_filepath, outputPixelType=sitk.sitkFloat32)))
img_ground_truth = tf.convert_to_tensor(img_raw, dtype=tf.float32)

rows, cols, depth = img_ground_truth.shape
channels = 1
pixel_count = rows * cols * depth
sampled_pixel_count = int(pixel_count * SAMPLING_RATIO)

img_mask, img_train = build_train_tensors(sampled_pixel_count, rows, cols, depth)

train_dataset = tf.data.Dataset.from_tensor_slices((img_mask, img_train))
train_dataset = train_dataset.shuffle(train_dataset.cardinality()).batch(BATCH_SIZE).cache()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

batch_size = min(BATCH_SIZE, len(img_mask))
num_steps = int(len(img_mask) * EPOCHS / batch_size)
print("Total training steps : ", num_steps)

checkpoint_dir = 'checkpoints/siren/inpainting/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
logdir = os.path.join('logs/siren/inpainting/', timestamp)
if not os.path.exists(logdir):
    os.makedirs(logdir)

run_training(epochs=EPOCHS, checkpoint_dir=checkpoint_dir, num_steps=num_steps)