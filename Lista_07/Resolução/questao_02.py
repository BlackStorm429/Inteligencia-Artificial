import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from PIL import Image

print("Versão do TensorFlow:", tf.__version__)
print("Versão do Keras:", tf.keras.__version__)

# Inicializando a Rede Neural Convolucional
classifier = Sequential()

# Primeira Camada de Convolução e Pooling
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Segunda Camada de Convolução e Pooling
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(Flatten())

# Dropout para evitar overfitting
classifier.add(Dropout(0.5))

# Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compilando a rede
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Pré-processamento das imagens de treino e validação
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset_personagens/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

validation_set = validation_datagen.flow_from_directory('dataset_personagens/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

# Ajustando steps_per_epoch e validation_steps conforme a quantidade de dados disponíveis
steps_per_epoch = training_set.samples // training_set.batch_size
validation_steps = validation_set.samples // validation_set.batch_size

# Convert DirectoryIterator to tf.data.Dataset and repeat
def convert_to_tf_dataset(directory_iterator):
    dataset = tf.data.Dataset.from_generator(
        lambda: directory_iterator,
        output_signature=(
            tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    return dataset.repeat()

training_dataset = convert_to_tf_dataset(training_set)
validation_dataset = convert_to_tf_dataset(validation_set)

# Prefetch data for performance optimization
training_dataset = training_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Treinando o modelo
classifier.fit(training_dataset,
               steps_per_epoch=steps_per_epoch,
               epochs=5,
               validation_data=validation_dataset,
               validation_steps=validation_steps)

# Função para prever a classe de uma imagem
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Imagem não encontrada: {image_path}")
        return
    test_image = Image.open(image_path)
    test_image = test_image.resize((64, 64))
    test_image = np.array(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    prediction = 'Homer' if result[0][0] > 0.5 else 'Bart'
    return prediction

# Lista de imagens para prever (exemplos)
image_paths = ['dataset_personagens/test_set/homer/homer1.bmp',
               'dataset_personagens/test_set/bart/bart1.bmp']

# Prevendo e mostrando as previsões para cada imagem
for image_path in image_paths:
    prediction = predict_image(image_path)
    if prediction:
        print(f"Imagem: {image_path}, Previsão: {prediction}")
        
        # Abrir a imagem usando PIL
        image = Image.open(image_path)
        
        # Mostrar a imagem
        image.show()
