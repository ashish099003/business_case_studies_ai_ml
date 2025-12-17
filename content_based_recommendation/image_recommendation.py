import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress absl "RAW" logs
os.environ["ABSL_LOG"] = "0"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
from keras.datasets import fashion_mnist



def image_dataset():
    (x_train, y_train), (x_test_org, y_test) = fashion_mnist.load_data()
    print("Training Data Shape:", x_train.shape)
    print("Test Data Shape:", x_test_org.shape)

if __name__=='__main__':
    image_dataset()