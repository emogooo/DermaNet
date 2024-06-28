from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

input_shape = (224, 224, 3)
batch_size = 16
epoch = 100

train_datagen = ImageDataGenerator(rescale=1.0 / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory("dataset/train", batch_size=batch_size, class_mode="categorical")
validation_set = test_datagen.flow_from_directory("dataset/validation", batch_size=batch_size, class_mode="categorical")
test_set = test_datagen.flow_from_directory("dataset/test", batch_size=10, class_mode="categorical", shuffle=False)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(GlobalAveragePooling2D())

model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=4, activation="softmax"))

model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

reduceLearningRate = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, min_lr=1e-4, verbose=1)

checkpoint = ModelCheckpoint(filepath="model_checkpoint.h5", monitor="val_loss", save_best_only=True, verbose=1)

history = model.fit(training_set, steps_per_epoch= training_set.samples//batch_size, epochs=epoch, validation_data=validation_set, validation_steps=validation_set.n // validation_set.batch_size, callbacks=[reduceLearningRate, checkpoint])

test_set.reset()
predictions = model.predict(test_set, steps=test_set.samples // test_set.batch_size)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())
conf_matrix = confusion_matrix(true_classes, predicted_classes)
accuracy = accuracy_score(true_classes, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix\nTotal Accuracy: {accuracy:.2f}")
plt.savefig("D:\Github\DK\ISIC\ESA\confusion_matrix")

fig, ax = plt.subplots()
ax.set_xlabel("Epoch", loc="right")
plt.title("Accuracy - Validation Accuracy")
plt.plot(history.history["accuracy"], "red", label="Accuracy")
plt.plot(history.history["val_accuracy"], "blue", label="Validation Accuracy")
plt.legend()
plt.savefig("D:\Github\DK\ISIC\ESA\\acc_val_acc_history")

fig, ax = plt.subplots()
ax.set_xlabel("Epoch", loc="right")
plt.title("Loss - Validation Loss")
plt.plot(history.history["loss"], "green", label="Loss")
plt.plot(history.history["val_loss"], "purple", label="Validation Loss")
plt.legend()
plt.savefig("D:\Github\DK\ISIC\ESA\loss_val_loss_history")
