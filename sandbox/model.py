import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

# Load pre-trained Inception-V3 model
model = InceptionV3(weights='imagenet')

# Load and preprocess the image
img_path = 'images/n01440764_tench.JPEG'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions
preds = model.predict(x)

# Decode the predictions
decoded_preds = decode_predictions(preds, top=3)[0]

# Print the top 3 predictions
for class_id, class_name, prob in decoded_preds:
    print(f'{class_name}: {prob * 100:.2f}%')

