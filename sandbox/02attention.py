import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
import cv2

# Load pre-trained Inception-V3 model
model = InceptionV3(weights='imagenet')

# Load and preprocess the image
img_path = 'images/n01440764_tench.JPEG'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Define the desired output layer for Grad-CAM
output_layer = model.layers[-1].output

# Create a new model that maps the input image to the desired output layer
gradcam_model = Model(inputs=model.input, outputs=[output_layer, model.get_layer('mixed10').output])

# Obtain the predictions and intermediate feature maps for the input image
output, feature_maps = gradcam_model.predict(x)

# Get the predicted class index
predicted_class_index = np.argmax(output[0])

# Obtain the gradient of the predicted class with respect to the intermediate feature maps
with tf.GradientTape() as tape1:
    features = tf.convert_to_tensor(feature_maps)
    tape1.watch(features)
    predictions = model.output[0, predicted_class_index]

grads1 = tape1.gradient(predictions, tf.identity(features))

with tf.GradientTape() as tape2:
    features = tf.convert_to_tensor(feature_maps)
    tape2.watch(features)
    predictions = model.output[0, predicted_class_index]

grads2 = tape2.gradient(predictions, tf.identity(features))

# Compute the channel-wise weights using global average pooling
channel_weights = tf.reduce_mean(tf.reduce_sum([grads1, grads2], axis=0), axis=(0, 1, 2))

# Compute the weighted combination of the feature maps
weighted_feature_maps = tf.reduce_sum(tf.multiply(channel_weights, feature_maps), axis=-1)

# Normalize the attention map
gradcam = np.maximum(weighted_feature_maps, 0)
gradcam /= np.max(gradcam)

# Resize the attention map to match the input image size
gradcam = cv2.resize(gradcam, (img.shape[1], img.shape[0]))

# Convert the attention map to a heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)

# Superimpose the heatmap on the original image
superimposed_img = cv2.addWeighted(cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

# Display the original image, attention map, and superimposed image
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')
plt.subplot(132)
plt.imshow(heatmap)
plt.title('Attention Map')
plt.axis('off')
plt.subplot(133)
plt.imshow(superimposed_img)
plt.title('Superimposed Image')
plt.axis('off')
plt.tight_layout()
plt.show()
