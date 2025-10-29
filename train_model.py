import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score
import seaborn as sns
from collections import Counter
from pathlib import Path


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.regularizers import l2

# Control memory growth to avoid OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # If GPU is available, limit memory growth
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        print("Memory growth setting failed")
else:
    print("\nRUNNING ON 'CPU'\n")
    # Set inter/intra parallelism threads for CPU
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

data_dir = r'D:\dataset_new' #path of the dataset

train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, 'test')

print("\nTrain data\n")
for dir in os.listdir(train_dir):
    full_path = os.path.join(train_dir, dir)
    file = list(Path(full_path).glob('*.jpg'))
    print(f'length of files {len(file)} in {dir}')

print("\nTest data\n")
for dir in os.listdir(test_dir):
    full_path = os.path.join(test_dir, dir)
    file = list(Path(full_path).glob('*.jpg'))
    print(f'length of files {len(file)} in {dir}')



# Function to show sample images from dataset
def show_image(data_dir):
    plt.figure(figsize=(15, 10))
    i = 1
    for dir in os.listdir(data_dir):
        full_path = os.path.join(data_dir, dir)
        if os.path.isdir(full_path):
            files = os.listdir(full_path)
            for file in files:
                if file.endswith('.jpg'):
                    img_path = os.path.join(full_path, file)
                    plt.subplot(2, 2, i)
                    image = plt.imread(img_path)
                    plt.imshow(image)
                    plt.title(f"Class: {dir}")
                    plt.axis('off')
                    i += 1
                    break
            if i > 4:  # Show max 4 images
                break
    plt.tight_layout()
    plt.show()

# Show sample images
show_image(train_dir)
show_image(test_dir)



train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    validation_split=0.2
)


test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Smaller batch size for CPU
BATCH_SIZE = 16
IMG_SIZE = 224

# Flow from directories
# Training data (80% of train_dir)
train_batches = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

# Validation data (20% of train_dir)
val_batches = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation', 
    shuffle=False,
    seed=42
)

test_batches = test_datagen.flow_from_directory(
    test_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False)

# Print class indices and counts
print("Train Classes:", Counter(train_batches.classes))
print("Test Classes:", Counter(test_batches.classes))
print("Val Classes:", Counter(val_batches.classes))

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Unfreeze after some initial training
for layer in base_model.layers[-50:]:
    layer.trainable = True


num_classes = len(np.unique(train_batches.classes))

x = base_model.output
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile with categorical crossentropy for multiclass classification
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)


callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3, min_lr=0.00001),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

STEPS_PER_EPOCH = min(len(train_batches), 500)
VALIDATION_STEPS = min(len(val_batches), 100)

history = model.fit(
    train_batches,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_batches,
    validation_steps=VALIDATION_STEPS,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

class_names = list(test_batches.class_indices.keys())

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_batches)
print(f"\n\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%\n\n")

# Predictions - use test_batches.reset() to ensure we're predicting on all test data
test_batches.reset()
print("Prediction\n")
y_pred_probs = model.predict(test_batches, steps=len(test_batches), verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_batches.classes

print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))



# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
# Save confusion matrix plot as image
plt.savefig('confusion_matrix.png', dpi=300)


# Calculate and print precision and recall for each class
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)

for idx, class_name in enumerate(class_names):
    print(f"{class_name} - Precision: {precision[idx]:.2f}, Recall: {recall[idx]:.2f}")


# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.savefig('accuracy_loss.png', dpi=300)
plt.tight_layout()
plt.show()
