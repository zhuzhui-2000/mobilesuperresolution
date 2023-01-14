from onnx_tf.backend import prepare
import onnx
import sys
import tensorflow as tf
from tensorflow.keras.models import save_model

if __name__ == '__main__':
    model = tf.keras.models.load_model("output_test.h5")
    model.summary()
