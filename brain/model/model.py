import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

def create_digit_model():
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(256, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(96, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    return model
    
def train_model():
    model = create_digit_model()
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    ck = tf.keras.callbacks.ModelCheckpoint(filepath="model_weights_mnist.hdf5", verbose=1, save_best_only=True)
    model.fit(x=x_train,y=y_train, validation_data=(x_test, y_test), epochs=100, callbacks=[es, ck])
    return model
    
def load_trained_model(weights_path):
    model = create_digit_model()
    model.load_weights(weights_path)
    return model