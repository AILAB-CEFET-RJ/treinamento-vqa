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

DATA_DIR = os.environ["DATA_DIR"]
FINAL_MODEL_FILE = os.path.join(DATA_DIR, "models", "inception-ft-best.h5")
TRIPLES_FILE = os.path.join(DATA_DIR, "triplas_imagenet_vqa.csv") 
IMAGE_DIR = DATA_DIR
IMAGENET_DIR = os.path.join(IMAGE_DIR, "ILSVRC", "Data", "DET", "train", "ILSVRC2013_train")
VQA_DIR = os.path.join(IMAGE_DIR, "vqa", "mscoco")

logger.debug("DATA_DIR %s", DATA_DIR)
logger.debug("FINAL_MODEL_FILE %s", FINAL_MODEL_FILE)
logger.debug("TRIPLES_FILE %s", TRIPLES_FILE)
logger.debug("IMAGE_DIR %s", IMAGE_DIR)

logger.debug("IMAGENET_DIR %s", IMAGENET_DIR)
logger.debug("VQA_DIR %s", VQA_DIR)
"""
logger.info("Carregando o modelo")
model = load_model(FINAL_MODEL_FILE)
logger.info("Modelo carregado com sucesso")
"""
logger.info("Iniciaando as predicoes")

for vqa_file in ["COCO_train2014_000000100337.jpg"]:
    logger.debug("Processando a imagem %s", vqa_file)
    
    image = Image.open(os.path.join(VQA_DIR, vqa_file)).convert("RGB")
    image = imresize(image, (299, 299))
    image = image.astype("float32")
    image = inception_v3.preprocess_input(image)
    X1 = np.expand_dims(image, axis=0)
    i = 1
    for synset_dir in ["n00007846"]:
        for imagenet_file in os.listdir(os.path.join(IMAGENET_DIR, synset_dir)):
            
            imagefile = os.path.join(IMAGENET_DIR, synset_dir, imagenet_file)
            print(imagefile)
            image2 = Image.open(imagefile).convert("RGB")
            image2 = imresize(image, (299, 299))
            image2 = image.astype("float32")
            image2 = inception_v3.preprocess_input(image2)
            X2 = np.expand_dims(image2, axis=0)

            logger.info("Carregando o modelo")
            model = load_model(FINAL_MODEL_FILE)
            logger.info("Modelo carregado com sucesso")

            y = model.predict([X1, X2])
            predicao = np.argmax(y, axis=1).tolist()
            
            print(y)
            
            print(vqa_file, imagenet_file, predicao)
            
            if i % 5 == 0:
                break
            else:
                i += 1
    
logger.info("Finalizado.")