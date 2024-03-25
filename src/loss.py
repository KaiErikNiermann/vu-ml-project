import keras
import config
from keras import ops

# Code is in part taken from https://www.kaggle.com/code/awsaf49/planttraits2024-kerascv-starter-notebook 
# and modified to account for imputing

class R2Loss(keras.losses.Loss):
    """
    R2 loss to train the model
    """
    def __init__(self, name="r2_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: true values of sample
            y_pred: predicted values of sample
        
        """
        SS_res = ops.sum(ops.square(y_true - y_pred), axis=0)  
        SS_tot = ops.sum(
            ops.square(y_true - ops.mean(y_true, axis=0)), axis=0
        )  
        r2_loss = SS_res / (SS_tot + 1e-6)  
        return ops.mean(r2_loss)  
    
class adjusted_R2Loss(keras.losses.Loss):
    """
    adjusted_R2Loss as alternative loss
    """
    def __init__(self, name="adjusted_r2_loss"):
        super().__init__(name=name)
        
    def call(self, y_true, y_pred):
        """
        Args:
            y_true: true values of sample
            y_pred: predicted values of sample
        """
        SS_res = ops.sum(ops.square(y_true - y_pred), axis=0)  
        SS_tot = ops.sum(
            ops.square(y_true - ops.mean(y_true, axis=0)), axis=0
        )  
        r2_loss = 1 - SS_res / (SS_tot + 1e-6)
        df_res = ops.shape(y_true)[0] - config.num_classes - 1
        df_tot = ops.shape(y_true)[0] - 1
        adjusted_r2_loss = 1 - (1 - r2_loss) * (df_tot / df_res)
        return ops.mean(adjusted_r2_loss)

class R2Metric(keras.metrics.Metric):
    """
    Use R2 metric to evaluate the model performance after training
    """
    def __init__(self, name="r2", **kwargs):
        super(R2Metric, self).__init__(name=name, **kwargs)
        self.SS_res = self.add_weight(name="SS_res", shape=(6,), initializer="zeros")
        self.SS_tot = self.add_weight(name="SS_tot", shape=(6,), initializer="zeros")
        self.num_samples = self.add_weight(name="num_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        SS_res = ops.sum(ops.square(y_true - y_pred), axis=0)
        SS_tot = ops.sum(ops.square(y_true - ops.mean(y_true, axis=0)), axis=0)
        self.SS_res.assign_add(SS_res)
        self.SS_tot.assign_add(SS_tot)
        self.num_samples.assign_add(ops.cast(ops.shape(y_true)[0], "float32"))

    def result(self):
        r2 = 1 - self.SS_res / (self.SS_tot + 1e-6)
        return ops.mean(r2)

    def reset_states(self):
        self.SS_res.assign(0)
        self.SS_tot.assign(0)
        self.num_samples.assign(0)
