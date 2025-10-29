import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

from eye_crop_best import detect_best_eye

# Load your trained model
model = load_model('best_model.h5') # Give path of the model

# Define class names
class_names = ['Closed', 'Open', 'no_yawn', 'yawn']

# Load and preprocess the image
image_path = r'testing_images\image.png' # path of the testing image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
input_image = np.expand_dims(image, axis=0)
input_image = preprocess_input(input_image)


eye = detect_best_eye(image_path, visualize=True)
eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
eye = cv2.resize(eye, (224, 224))
input_eye = np.expand_dims(eye, axis=0)
input_eye = preprocess_input(input_eye)

# Predict
pred = model.predict(input_image)
predicted_class = class_names[np.argmax(pred)]
confidence = np.max(pred)

pred_eye = model.predict(input_eye)
predicted_class_eye = class_names[np.argmax(pred_eye)]
confidence_eye = np.max(pred_eye)

# Show result
print(f"Predicted class of the image: {predicted_class} and {predicted_class_eye} ({((confidence+confidence_eye)/2):.2f} confidence)")
if predicted_class=="no_yawn" and predicted_class_eye=="Closed":
    state = "Drowsy"
    print(state)
elif predicted_class=="yawn" and predicted_class_eye=="Closed":
    state = "Drowsy"
    print(state)
elif predicted_class=="yawn" and predicted_class_eye=="Open":
    state = "Drowsy"
    print(state)
elif predicted_class=="yawn" and predicted_class_eye=="yawn":
    state = "Drowsy"
    print(state)
else:
    state = "Non-Drowsy"
    print(state)


plt.imshow(image)
plt.title(f"{predicted_class} and {predicted_class_eye} ({((confidence+confidence_eye)/2):.2f}) , State={state}")
plt.axis('off')
plt.show()
