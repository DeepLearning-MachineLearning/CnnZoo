from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
        Conv2D, MaxPooling2D, Flatten, Input, 
        Concatenate, AveragePooling2D,
        Dropout, Softmax, Dense)

def GoogLeNet(input_shape=(224,224,3), is_training=False):
    def inception_block(num_1x1, num_3x3_reduc, num_3x3, num_5x5_reduc, num_5x5, num_pool_proj, x):
        b1 = Conv2D(num_1x1,1,strides=1,padding="same",activation="relu")(x)
        b2 = Conv2D(num_3x3_reduc,1,strides=1,padding="same", activation="relu")(x)
        b2 = Conv2D(num_3x3, 3,strides=1,padding="same", activation="relu")(b2)
        b3 = Conv2D(num_5x5_reduc,1,strides=1,padding="same", activation="relu")(x)
        b3 = Conv2D(num_5x5,5,strides=1,padding="same", activation="relu")(b3)
        b4 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding="same")(x)
        b4 = Conv2D(num_pool_proj,1,strides=1,padding="same", activation="relu")(b4)
        #suppose the input is HWC format
        x = Concatenate(axis=-1)([b1,b2,b3,b4])
        return x

    def softmax_head(x):
        '''Softmax head to help the traning
        '''
        y = AveragePooling2D(pool_size=(5,5),strides=3)(x)
        y = Conv2D(1024, 1, strides=1, padding="same", activation="relu")(y)
        y = Flatten()(y)
        y = Dense(1000)(y)
        y = Dense(1000)(y)
        y = Softmax()(y)
        return y

    outputs = []
    inputs = Input(shape=input_shape)
    x = Conv2D(64,7, strides=2, padding="same",activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same")(x)
    x = Conv2D(192,3, strides=1, padding="same",activation="relu")(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same")(x)
    x = inception_block(64,96,128,16,32,32,x)
    x = inception_block(128,128,192,32,96,64,x)
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same")(x)
    x = inception_block(192,96,208,16,48,64,x)
    if is_training: outputs.append(softmax_head(x))
    x = inception_block(160,112,224,24,64,64,x)
    x = inception_block(128,128,256,24,64,64,x)
    x = inception_block(112,144,288,32,64,64,x)
    if is_training: outputs.append(softmax_head(x))
    x = inception_block(256,160,288,32,64,64,x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same")(x)
    x = inception_block(256,160,288,32,64,64,x)
    x = inception_block(384,192,384,48,128,128,x)
    x = AveragePooling2D(pool_size=(7,7),strides=1)(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(1000)(x)
    x = Softmax()(x)
    outputs.append(x)
    return  Model(inputs=inputs,outputs=outputs)

def google_net(input_shape=(224,224,3)):
    return  GoogLeNet(input_shape)

if __name__ == "__main__":
    GoogLeNet((448,448,3), is_training=True).summary()
    GoogLeNet((448,448,3), is_training=False).summary()
