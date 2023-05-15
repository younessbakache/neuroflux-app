import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import numpy as np

class BaseModel:
    def __init__(self, train_dir, test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model = None
        self.class_labels = None

    def load_data(self):
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        validation_datagen = ImageDataGenerator(rescale=1./255)

        train_set = train_datagen.flow_from_directory(self.train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
        validation_set = validation_datagen.flow_from_directory(self.test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

        self.class_labels = train_set.class_indices
        self.class_labels = dict((v,k) for k,v in self.class_labels.items())

        return train_set, validation_set

    def evaluate(self, model_path):
        self.model = load_model(model_path)
        _, validation_set = self.load_data()
        loss, accuracy = self.model.evaluate(validation_set)
        print(f'Test loss: {loss}, Test accuracy: {accuracy}')

    def predict(self, model_path, image_path):
        if self.model is None:
            self.model = load_model(model_path)
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.

        predictions = self.model.predict(x)
        predicted_class = np.argmax(predictions)
        predicted_label = self.class_labels[predicted_class]
        print("Neuroflux disorder phase of the given MRI Scan is: " + predicted_label)


class Model1(BaseModel):
    def __init__(self, train_dir, test_dir):
        super().__init__(train_dir, test_dir)

    def train(self, model_path):
        train_set, validation_set = self.load_data()

        base_model = ResNet50(weights='imagenet', include_top=False)
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(5, activation='softmax')(x) 

        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(train_set, validation_data=validation_set, epochs=10)
        self.model.save(model_path)


class Model2(BaseModel):
    def __init__(self, train_dir, test_dir):
        super().__init__(train_dir, test_dir)

    def train(self, model_path):
        train_set, validation_set = self.load_data()

        self.model = tf.keras.models.Sequential([
            Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(5, activation='softmax')
        ])

        self.model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(train_set, validation_data=validation_set, epochs=10)
        self.model.save(model_path)
import os
import shutil
import numpy as np

# Define a function to split the data
def split_data(source_dir, train_dir, test_dir, split_size):
    files = []
    for filename in os.listdir(source_dir):
        file = source_dir + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")
    
    training_length = int(len(files) * split_size)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]
    
    for filename in training_set:
        this_file = source_dir + filename
        destination = train_dir + filename
        shutil.copyfile(this_file, destination)
    
    for filename in testing_set:
        this_file = source_dir + filename
        destination = test_dir + filename
        shutil.copyfile(this_file, destination)
        


model1 = Model1('/data/train/', '/data/test/')
model1.train('model1.h5')
model1.evaluate('model1.h5')
model1.predict('model1.h5', '/data/test/EO/neuroflux_003_S_6258_MR_Axial_T2_STAR__br_raw_20190502124835890_22_S820673_I1160992.jpg')

model2 = Model2('/data/train/', 'data/test/')
model2.train('model2.h5')
model2.evaluate('model2.h5')
model2.predict('model2.h5', '/data/test/EO/neuroflux_003_S_6258_MR_Axial_T2_STAR__br_raw_20190502124835890_22_S820673_I1160992.jpg')
