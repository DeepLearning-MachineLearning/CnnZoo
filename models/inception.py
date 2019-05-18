from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
        Conv2D, MaxPooling2D, Flatten, Input, 
        Concatenate, AveragePooling2D,
        Dropout, Softmax, Dense, Add, Lambda)

def GoogLeNet(input_shape=(224,224,3), class_num=1000, is_training=False):
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
    x = Dense(class_num)(x)
    x = Softmax()(x)
    outputs.append(x)
    return  Model(inputs=inputs,outputs=outputs)

def inception_resnet_v2(input_shape=(224,224,3), class_num=1000):
    '''Google's Inception-ResNet v2 model
    '''
    def inception_block(inputs, branches, activation="relu"):
        branch_tensors = []
        for i, branch in enumerate(branches):
            branch_tensors.append([])
            x = inputs
            for layer in branch:
                ltype, args = layer[0], layer[1:]
                if ltype == "Conv2D":
                    x = Conv2D(*args, activation=activation)(x)
                elif ltype == "MaxPooling2D":
                    x = MaxPooling2D(*args)(x)
                else:
                    raise ValueError ("Invalid layer type" + ltype)
                branch_tensors[i].append(x)
        CONCAT_DIM = -1
        #concat the last tensor of each branch in C channel
        x = Concatenate(CONCAT_DIM)([branch[-1] for branch in branch_tensors])
        return x

    def inception_v4_sa(x):
        branches = [ [("MaxPooling2D",3,2,"valid")],
                     [("Conv2D",96,3,2,"valid")]]
        return inception_block(x, branches)

    def inception_v4_sb(x):
        branches =  [ [("Conv2D",64,1,1, "same"), ("Conv2D",96,3,1,"valid")],
                      [("Conv2D",64,1,1, "same"), ("Conv2D",64,(7,1),1,"same"), ("Conv2D", 64, (1,7), 1, "same"), ("Conv2D",96,3,1, "valid")]]
        return inception_block(x, branches)

    def inception_v4_sc(x):
        branches = [ [("Conv2D",192,3,2, "valid"),],
                     [("MaxPooling2D", 3,2,"valid")]]
        return inception_block(x, branches)

    def inception_resnet_v2_a(x):
        branches = [ [("Conv2D",31,1,1,"same")],
                     [("Conv2D",32,1,1,'same'), ('Conv2D', 32,3,1,'same')],
                     [('Conv2D',32,1,1,'same'), ('Conv2D', 48,3,1,'same'),('Conv2D', 64,3,1,'same')]]
        x = inception_block(x, branches)
        x = Conv2D(384,1,1,'same')(x)
        return x
    
    def inception_resnet_v2_b(x):
        #TODO: confirm 128->160->192 is right for 1x7->7x1 branch?
        branches = [[('Conv2D',192,1,1,'same')],
                    [('Conv2D',128,1,1,'same'),('Conv2D',160,(1,7),1,'same'),('Conv2D',192,(7,1),1,'same')]]
        x = inception_block(x, branches)
        #TODO: Is this correct?
        x = Conv2D(1152,1,1,'same')(x)
        return x

    def inception_resnet_v2_c(x):
        branches = [ [("Conv2D",192,1,1,"same")],
                     [("Conv2D",192,1,1,'same'), ('Conv2D', 224, (1,3),1,'same'), ('Conv2D',256,(3,1),1,'same')]]
        x = inception_block(x, branches)
        x = Conv2D(2048,1,1,'same')(x)
        return x

    def inception_v4_ra(x):
        branches = [ [('MaxPooling2D',3,2,'valid')],
                     [('Conv2D',384,3,2,'valid')],
                     [('Conv2D',256,1,1,'same'),('Conv2D',256,3,1,'same'),('Conv2D',384,3,2,'valid')]]
        return inception_block(x, branches)

    def inception_resnet_v2_rb(x):
        branches = [[('MaxPooling2D', 3,2,'valid')],
                    [('Conv2D',256,1,1,'same'),('Conv2D',384,3,2,'valid')],
                    [('Conv2D',256,1,1,'same'),('Conv2D',256,3,2,'valid')],
                    [('Conv2D',256,1,1,'same'),('Conv2D',256,3,1,'same'),('Conv2D',256,3,2,'valid')]]
        return inception_block(x, branches)

    def residual_block(x, flow, scale=1.0, activation=None):
        outputs = flow(x)
        #TODO: support NCHW
        #asumming it's NHWC
        strides = ((x.shape[-3]-1)//outputs.shape[-3]+1,
                   (x.shape[-2]-1)//outputs.shape[-2]+1)
        if x.shape != outputs.shape or strides != (1,1):
            # Use Conv2D to make the shape concistent, no activation
            x = Conv2D(outputs.shape.as_list()[-1], 1, strides, "same")(x)

        def Scale(i):
            return i*scale
        x = Add()([x, Lambda(Scale)(outputs)])
        if activation:
            x = activation(x)
        return x

    outputs = []
    inputs = Input(shape=input_shape)
    x = Conv2D(32,3, strides=2, padding="valid",activation="relu")(inputs)
    x = Conv2D(32,3, strides=1, padding="valid",activation="relu")(x)
    x = Conv2D(64,3, strides=1, padding="same",activation="relu")(x)

    residual_scale = 0.2
    x = inception_v4_sa(x)
    x = inception_v4_sb(x)
    x = inception_v4_sc(x)
    for _ in range(5):
        x = residual_block(x, inception_resnet_v2_a, scale=residual_scale)
    x = inception_v4_ra(x)
    for _ in range(10):
        x = residual_block(x, inception_resnet_v2_b, scale=residual_scale)
    x = inception_resnet_v2_rb(x)
    for _ in range(5):
        x = residual_block(x, inception_resnet_v2_c, scale=residual_scale)
    x = AveragePooling2D()(x)
    x = Dropout(0.8)(x)
    
    # This head can be changed
    x = Flatten()(x)
    x = Dense(class_num)(x)
    x = Softmax()(x)

    outputs.append(x)
    return  Model(inputs=inputs,outputs=outputs)


def google_net(input_shape=(224,224,3)):
    return  GoogLeNet(input_shape)

if __name__ == "__main__":
    GoogLeNet((448,448,3), is_training=True).summary()
    GoogLeNet((448,448,3), is_training=False).summary()
    inception_resnet_v2((299,299,3)).summary()
