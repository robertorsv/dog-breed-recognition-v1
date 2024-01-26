import cv2
import numpy as np 
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model,Model
from keras.optimizers import RMSprop
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,BatchNormalization
from keras.applications.resnet_v2 import ResNet50V2,preprocess_input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from PIL import Image

num_breeds = 60
im_size = 224
batch_size = 64
encoder = LabelEncoder()

df_labels = pd.read_csv("labels.csv")
train_file = 'train/'
test_file = 'test/'

breed_dict = list(df_labels['breed'].value_counts().keys()) 
new_list = sorted(breed_dict,reverse=True)[:num_breeds*2+1:2]
df_labels = df_labels.query('breed in @new_list')
df_labels['img_file'] = df_labels['id'].apply(lambda x: x + ".jpg")

train_x = np.zeros((len(df_labels), im_size, im_size, 3), dtype='float32')

for i, img_id in enumerate(df_labels['img_file']):
    img = Image.open(train_file+img_id)
    img = img.resize((im_size, im_size))
    img_array = preprocess_input(np.expand_dims(np.array(img).astype(np.float32), axis=0))
    train_x[i] = img_array

train_y = encoder.fit_transform(df_labels["breed"].values)

x_train, x_test, y_train, y_test = train_test_split(train_x,train_y,test_size=0.2,random_state=42)

train_datagen = ImageDataGenerator(rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.25,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow(x_train, 
                                     y_train, 
                                     batch_size=batch_size)

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow(x_test, 
                                   y_test, 
                                   batch_size=batch_size)

resnet = ResNet50V2(input_shape = [im_size,im_size,3], weights='imagenet', include_top=False)
for layer in resnet.layers:
    layer.trainable = False

x = resnet.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(num_breeds, activation='softmax')(x)

model = Model(inputs=resnet.input, outputs=predictions)

epochs = 20
learning_rate = 1e-3

optimizer = RMSprop(learning_rate=learning_rate,rho=0.9)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(train_generator,
          steps_per_epoch= x_train.shape[0] // batch_size,
          epochs= epochs,
          validation_data= test_generator,
          validation_steps= x_test.shape[0] // batch_size,
          callbacks=[reduce_lr, early_stop])

model.save("model")

model = load_model("model")

pred_img_path = 'germanshepherd.jpg'
pred_img_array = Image.open(pred_img_path)
pred_img_array = pred_img_array.resize((im_size, im_size))
pred_img_array = preprocess_input(np.expand_dims(np.array(pred_img_array).astype(np.float32), axis=0))

pred_label = model.predict_step(np.array(pred_img_array,dtype="float32"))

pred_label = np.argmax(pred_label, axis=1)

pred_breed = encoder.inverse_transform(pred_label)
print("Predicted Breed for this Dog is :",pred_breed)
