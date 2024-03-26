import keras_cv
import tensorflow.keras as keras
from tensorflow.keras import layers
from loss import R2Loss, R2Metric
from keras.metrics import Accuracy
import config

def Model(config, slct_backbone):
    backbones = {
        "efficientnetv2_b2_imagenet": keras_cv.models.EfficientNetV2B2Backbone,
        "efficientnetv2_s_imagenet": keras_cv.models.EfficientNetV2SBackbone,
        "resnet50_imagenet": keras_cv.models.ResNet50Backbone,
    }

    # Define input layer
    img_input = keras.Input(shape=(*config.image_size, 3), name="images")

    # Branch for image input
    backbone = backbones[slct_backbone](slct_backbone)
    backbone_output = backbone(img_input)
    gavrg1 = layers.GlobalAveragePooling2D()(backbone_output)
    dense1 = layers.Dense(1024, activation="relu")(gavrg1)
    dense2 = layers.Dense(512, activation="relu")(dense1)
    dropout = layers.Dropout(0.2)(dense2)

    # Output layers
    head_out = layers.Dense(config.num_classes, activation=None, name="head")(dropout)
    aux_head_out = layers.Dense(config.aux_num_classes, activation="relu", name="aux_head")(dropout)

    model = keras.Model(inputs=img_input, outputs={"head": head_out, "aux_head": aux_head_out})

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
