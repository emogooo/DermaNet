import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, MultiHeadAttention, LayerNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications import DenseNet201

# Transformer Encoder Layer tanımı
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, embed_dim, dense_dim, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(dense_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Transformer Decoder Layer tanımı
class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, embed_dim, dense_dim, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(dense_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x, enc_output, training):
        attn1 = self.mha1(x, x)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        
        attn2 = self.mha2(out1, enc_output)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)
    
    
def create_cnn_transformer_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # CNN katmanları
    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    
    # Özelliklerin Transformer Encoder'a uygun forma getirilmesi
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = tf.expand_dims(x, axis=1)  # Transformer için bir boyut eklenir
    
    # Transformer Encoder katmanı
    encoder_layer = TransformerEncoderLayer(num_heads=4, embed_dim=128, dense_dim=256)
    enc_output = encoder_layer(x)
    enc_output = tf.squeeze(enc_output, axis=1) 
    
    """
    # Transformer Decoder katmanı
    decoder_layer = TransformerDecoderLayer(num_heads=4, embed_dim=128, dense_dim=256)
    dec_output = decoder_layer(enc_output, enc_output)
    dec_output = tf.squeeze(dec_output, axis=1)  # Eklenen boyut kaldırılır
    """
    
    # Fully Connected katmanları
    x = Dense(256, activation="relu")(enc_output)
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)
    
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    
    return model

def create_transfer_model(input_shape, num_classes):
    transfer_model = DenseNet201(weights= 'imagenet', include_top = False, input_shape=input_shape)

    for layer in transfer_model.layers[:-10]:
        layer.trainable = False
    
    new_model = Sequential()
    new_model.add(GlobalAveragePooling2D(input_shape = transfer_model.output_shape[1:], data_format=None))
    new_model.add(Dense(256, activation='relu'))
    new_model.add(Dropout(0.3))
    new_model.add(Dense(128, activation='relu'))
    new_model.add(Dropout(0.6))
    new_model.add(Dense(num_classes, activation='softmax'))

    model = Model(inputs=transfer_model.input, outputs=new_model(transfer_model.output))
    
    return model
    

target_size = (224, 224)
input_shape = (224, 224, 3)
batch_size = 20

num_classes = 4
class_names = ['BCC', 'BKL', 'MEL', 'NV']
class_indices = {name: idx for idx, name in enumerate(class_names)}

class_weights = {
    class_indices['BCC']: 0.95,  # BCC için ağırlık
    class_indices['BKL']: 1.25,  # BKL için ağırlık
    class_indices['MEL']: 1.1,  # MEL için ağırlık
    class_indices['NV']: 1.35    # NV için ağırlık
}

train_datagen = ImageDataGenerator(rescale=1.0 / 128, rotation_range=40, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
test_datagen = ImageDataGenerator(rescale=1.0 / 128)

training_set = train_datagen.flow_from_directory(
    "dataset/train", target_size=target_size, batch_size=batch_size, class_mode="categorical")
validation_set = test_datagen.flow_from_directory(
    "dataset/validation", target_size=target_size, batch_size=batch_size, class_mode="categorical")
test_set = test_datagen.flow_from_directory(
    "dataset/test", target_size=target_size, batch_size=batch_size, class_mode="categorical")



#model = create_cnn_transformer_model(input_shape, num_classes)

model = create_transfer_model(input_shape, num_classes)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

history = model.fit(
    training_set,
    steps_per_epoch=training_set.n // training_set.batch_size,
    epochs=15,
    validation_data=validation_set,
    validation_steps=validation_set.n // validation_set.batch_size,
    class_weight=class_weights
)

# Test verisi üzerinde tahminler
test_set.reset()
predictions = model.predict(test_set, steps=test_set.n // test_set.batch_size)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())

# Test verisi üzerinde tahminler
test_set.reset()
predictions = model.predict(test_set, steps=test_set.n // test_set.batch_size)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_set.classes
print(len(predicted_classes))
class_labels = list(test_set.class_indices.keys())

# Konfizyon matrisi oluşturma
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Toplam doğruluk (accuracy) hesaplama
accuracy = accuracy_score(true_classes, predicted_classes)

# Konfizyon matrisini görselleştirme ve kaydetme
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix\nTotal Accuracy: {accuracy:.2f}')
plt.savefig("D:\Github\DK\ISIC\ESA\confusion_matrix1")

# Sınıflandırma raporu
print(classification_report(true_classes,
      predicted_classes, target_names=class_labels))

fig, ax = plt.subplots()
ax.set_xlabel("Epoch", loc="right")
plt.title("Accuracy - Validation Accuracy")
plt.plot(history.history["accuracy"], "red", label="Accuracy")
plt.plot(history.history["val_accuracy"], "blue", label="Validation Accuracy")
plt.legend()
plt.savefig("D:\Github\DK\ISIC\ESA\\acc_val_acc_history1")

fig, ax = plt.subplots()
ax.set_xlabel("Epoch", loc="right")
plt.title("Loss - Validation Loss")
plt.plot(
    history.history["loss"],
    "green",
    label="Loss"
)
plt.plot(history.history["val_loss"], "purple", label="Validation Loss")
plt.legend()
plt.savefig("D:\Github\DK\ISIC\ESA\loss_val_loss_history1")
