import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_siren import siren_mlp
import SimpleITK as sitk

BATCH_SIZE = 8192

# Image - CT scan
def normalize_array(arr):
    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return norm_arr

img_filepath = 'data/image_ct.mha'
img_raw = normalize_array(sitk.GetArrayFromImage(sitk.ReadImage(img_filepath, outputPixelType=sitk.sitkFloat32)))
img_ground_truth = tf.convert_to_tensor(img_raw, dtype=tf.float32)

rows, cols, depth = img_ground_truth.shape
channels = 1
pixel_count = rows * cols * depth


def build_eval_tensors(channels):
    img_mask_x = tf.range(0, rows, dtype=tf.int32)
    img_mask_y = tf.range(0, cols, dtype=tf.int32)
    img_mask_z = tf.range(0, depth, dtype=tf.int32)

    img_mask_x, img_mask_y, img_mask_z = tf.meshgrid(img_mask_x, img_mask_y, img_mask_z, indexing='ij')

    img_mask_x = tf.expand_dims(img_mask_x, axis=-1)
    img_mask_y = tf.expand_dims(img_mask_y, axis=-1)
    img_mask_z = tf.expand_dims(img_mask_z, axis=-1)

    img_mask_x = tf.cast(img_mask_x, tf.float32) / rows
    img_mask_y = tf.cast(img_mask_y, tf.float32) / cols
    img_mask_z = tf.cast(img_mask_z, tf.float32) / depth

    img_mask = tf.concat([img_mask_x, img_mask_y, img_mask_z], axis=-1)
    img_mask = tf.reshape(img_mask, [-1, 3])

    img_train = tf.reshape(img_ground_truth, [-1, channels])

    return img_mask, img_train


img_mask, img_eval = build_eval_tensors(channels)

test_dataset = tf.data.Dataset.from_tensor_slices((img_mask, img_eval))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Build model
model = siren_mlp.SIRENModel(units=256, final_units=channels, final_activation='sigmoid', num_layers=5, w0=1.0, w0_initial=30.0)

# Restore model
checkpoint_path = 'checkpoints/siren/inpainting/model'
if len(glob.glob(checkpoint_path + "*.index")) == 0:
    raise FileNotFoundError("Model checkpoint not found !")

# instantiate model
_ = model(tf.zeros([1, 3]))

# load checkpoint
model.load_weights(checkpoint_path + '.weights.h5')

predicted_image = model.predict(test_dataset, batch_size=BATCH_SIZE, verbose=1)
predicted_image = predicted_image.reshape((rows, cols, depth))  # type: np.ndarray
predicted_image = predicted_image.clip(0.0, 1.0)

sitk.WriteImage(sitk.GetImageFromArray(predicted_image), 'data/image_predicted.mha')
sitk.WriteImage(sitk.GetImageFromArray(img_ground_truth), 'data/image_gt.mha')

# fig, axes = plt.subplots(1, 2)
# plt.sca(axes[0])
# plt.imshow(img_ground_truth.numpy()[70,:,:])
# plt.title("Ground Truth Image")

# plt.sca(axes[1])
# plt.imshow(predicted_image[70,:,:])
# plt.title("Predicted Image")

# fig.tight_layout()
# plt.show()
