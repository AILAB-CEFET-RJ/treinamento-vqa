import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imresize
from keras.applications import inception_v3
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix
#################################################################
#               Configurando logs de execucao                   #
#################################################################
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/predict.log',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


logger.info("Iniciando...")
logger.info("FInalizado.")