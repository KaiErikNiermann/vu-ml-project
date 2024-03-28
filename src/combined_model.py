import keras_cv
import tensorflow.keras as keras
from tensorflow.keras import layers
from loss import R2Loss, R2Metric
from keras.metrics import Accuracy
import config

def cModel(config, feature_cols, img_model, model_name=None):
    
    feature_in = keras.Input(shape=(len(feature_cols),), name="features")
    
    # tabular input branch
    dense = layers.Dense(326, activation="selu")(feature_in)
    dense = layers.Dense(64, activation="selu")(dense)
    dropout = layers.Dropout(0.1)(dense)
    
    concat = layers.Concatenate()([img_model[0], dropout])
    
    # output layers
    head_out = layers.Dense(config.num_classes, activation=None, name="head")(concat)
    aux_head_out = layers.Dense(config.aux_num_classes, activation="relu", name="aux_head")(concat)

    model = keras.models.Model(inputs=[img_model[1], feature_in], outputs={"head": head_out, "aux_head": aux_head_out})
    
    if model_name:
        model.name = model_name

    # Compile model
    print("Compiling model")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss={"head": R2Loss(), "aux_head": R2Loss()},
        loss_weights={"head": 1.0, "aux_head": 0.3},
        metrics={"head": R2Metric()},
    )

    return model

def model_summary(model):
    model.build((None, *config.image_size, 3))
    model.summary()
