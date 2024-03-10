# deep learning
import keras_cv
import keras
from keras import layers
from loss import *

class Model(keras.Model):
    def __init__(self, config, slct_backbone):
        super(Model, self).__init__()
        backbones = {
            "efficientnetv2_b2_imagenet": keras_cv.models.EfficientNetV2B2Backbone,
            "efficientnetv2_s_imagenet": keras_cv.models.EfficientNetV2SBackbone,
            "resnet50_imagenet": keras_cv.models.ResNet50Backbone,
        }

        # Define input layer
        self.img_input = keras.Input(shape=(*config.image_size, 3), name="images")

        # Branch for image input
        backbone = backbones[slct_backbone](slct_backbone)
        x1 = backbone(self.img_input)
        x1 = layers.GlobalAveragePooling2D()(x1)
        x1 = layers.Dense(512, activation="relu")(x1)
        x1 = layers.Dropout(0.2)(x1)

        # Output layer
        out1 = layers.Dense(config.num_classes, activation=None, name="head")(x1)
        out2 = layers.Dense(config.aux_num_classes, activation="relu", name="aux_head")(x1)

        self.model = keras.Model(inputs=self.img_input, outputs={"head": out1, "aux_head": out2})

    def compile_model(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss={"head": R2Loss(), "aux_head": R2Loss()},
            loss_weights={"head": 1.0, "aux_head": 0.4},
            metrics={"head": R2Metric()}
        )

    def summary(self):
        self.model.summary()