import matplotlib.pyplot as plt
import os
from keras import layers
from keras import Model
from keras.applications.inception_v3 import InceptionV3

train_dir = r"D:\thetest\Railway-Track-fault-Detection-Project-main\content\Railway Track fault Detection Updated\Train"
validation_dir = r"D:\thetest\Railway-Track-fault-Detection-Project-main\content\Railway Track fault Detection Updated\Validation"

# Directory with our training defective/nondefective pictures
train_defective_dir = r"D:\thetest\Railway-Track-fault-Detection-Project-main\content\Railway Track fault Detection Updated\Train\Defective"
train_nondefective_dir = r"D:\thetest\Railway-Track-fault-Detection-Project-main\content\Railway Track fault Detection Updated\Train\Non Defective"

# Directory with our validation defective/nondefective pictures
validation_defective_dir = r"D:\thetest\Railway-Track-fault-Detection-Project-main\content\Railway Track fault Detection Updated\Validation\Defective"
validation_nondefective_dir = r"D:\thetest\Railway-Track-fault-Detection-Project-main\content\Railway Track fault Detection Updated\Validation\Non Defective"


train_defective_fnames = os.listdir(train_defective_dir )
train_nondefective_fnames = os.listdir( train_nondefective_dir)


local_weights_file = r"D:\thetest\Railway-Track-fault-Detection-Project-main\content\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

pre_trained_model = InceptionV3(input_shape = (300,300, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False
# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


#data preprocessing
from keras.preprocessing.image import ImageDataGenerator


# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(300,300))     
# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (300,300))


# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(32, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = 'Adam', 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])


history = model.fit(train_generator,
                              validation_data=validation_generator,
                              epochs=25,                            
                              verbose=2)

model_save_name=model.save('mymodel.h5',history)
print('model save successfully')

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.savefig(r"D:\thetest\Railway-Track-fault-Detection-Project-main\work_dir\accuracy.png")
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'  )
plt.savefig(r"D:\thetest\Railway-Track-fault-Detection-Project-main\work_dir\loss.png")