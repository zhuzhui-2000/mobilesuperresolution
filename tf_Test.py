
# import keras
# import tensorflow as tf
# from PIL import Image
import torch

if __name__ == '__main__':
    # model = keras.models.load_model("output_test.h5")
    # model.summary()
    # interpreter = tf.lite.Interpreter(model_path='/home/zhuzhui/super-resolution/MyNAS/compiler-aware-nas-sr/tflite_model/test_model.tflite')
    # interpreter.allocate_tensors()
    # image = Image.open(img).convert('RGB')
    model = torch.load('/home/zhuzhui/super-resolution/MyNAS/compiler-aware-nas-sr/runs/wdsr_b_x2_16_24_Sep09_22_53_35/weights/models.pt')
    print(model)
