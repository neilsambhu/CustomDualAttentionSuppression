import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

# Load the InceptionV3 model pre-trained on ImageNet
model = InceptionV3(weights='imagenet')

# Load and preprocess the input image
img_path = 'images/n01440764_tench.JPEG'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions on the image
preds = model.predict(x)
predicted_class = np.argmax(preds[0])
predicted_class_name = decode_predictions(preds, top=1)[0][0][1]

# Get the output tensor of the last convolutional layer in the model
last_conv_layer = model.get_layer('mixed10')

# Create a model that outputs both the predictions and the output tensor of the last conv layer
grad_model = tf.keras.models.Model(inputs=model.input, outputs=(model.output, last_conv_layer.output))

# Compute the gradient of the predicted class with respect to the output tensor of the last conv layer
with tf.GradientTape() as tape:
    preds, last_conv_output = grad_model(x)
    class_output = preds[:, predicted_class]
grads = tape.gradient(class_output, last_conv_output)

# Compute the channel-wise mean of the gradients
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# Multiply each channel in the feature map array by the corresponding gradient value
last_conv_output = last_conv_output[0]
heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# Apply ReLU to the heatmap
heatmap = tf.nn.relu(heatmap)

# Normalize the heatmap
heatmap /= tf.reduce_max(heatmap)

# Resize the heatmap to the size of the original image
heatmap = tf.image.resize(heatmap, (tf.shape(img)[1], tf.shape(img)[0]))

# Convert the heatmap to RGB
heatmap = tf.expand_dims(heatmap, axis=-1)
heatmap_rgb = tf.image.grayscale_to_rgb(heatmap)

# Superimpose the heatmap on the original image
superimposed_img = heatmap_rgb * 0.4 + img

# Plot the original image, heatmap, and superimposed image
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[1].imshow(heatmap, cmap='jet')
axes[1].set_title('Attention Heatmap')
axes[2].imshow(superimposed_img / 255.0)
axes[2].set_title('Superimposed Image')
plt.tight_layout()
plt.show()
