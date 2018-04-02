import os

from keras import backend as K
from keras.applications import inception_v3
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, merge
from keras.layers.core import Activation, Dense, Dropout, Lambda
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from random import shuffle
from scipy.misc import imresize
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

#################################################################
#               Configurando logs de execucao                   #
#################################################################
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/training.log',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

#################################################################
#                           Métodos                             #
#################################################################
def carregar_triplas(lista_triplas):
    image_triples = pd.read_csv(lista_triplas, sep=",", header=0, names=["left","right","similar"])
    return image_triples.values
#################################################################
def load_image_cache(image_cache, image_filename):
    image = plt.imread(os.path.join(IMAGE_DIR, image_filename))
    image = imresize(image, (299, 299))
    image = image.astype("float32")
    image = inception_v3.preprocess_input(image)
    image_cache[image_filename] = image

#################################################################
def pair_generator(triples, image_cache, datagens, batch_size=32):
    while True:
        # shuffle once per batch
        indices = np.random.permutation(np.arange(len(triples)))
        num_batches = len(triples) // batch_size
        for bid in range(num_batches):
            batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]
            batch = [triples[i] for i in batch_indices]
            X1 = np.zeros((batch_size, 299, 299, 3))
            X2 = np.zeros((batch_size, 299, 299, 3))
            Y = np.zeros((batch_size, 2))
            for i, (image_filename_l, image_filename_r, label) in enumerate(batch):
                if datagens is None or len(datagens) == 0:
                    X1[i] = image_cache[image_filename_l]
                    X2[i] = image_cache[image_filename_r]
                else:
                    X1[i] = datagens[0].random_transform(image_cache[image_filename_l])
                    X2[i] = datagens[1].random_transform(image_cache[image_filename_r])
                Y[i] = [1, 0] if label == 0 else [0, 1]
            yield [X1, X2], Y

#################################################################
def cosine_distance(vecs, normalize=False):
    x, y = vecs
    if normalize:
        x = K.l2_normalize(x, axis=0)
        y = K.l2_normalize(x, axis=0)
    return K.prod(K.stack([x, y], axis=1), axis=1)

#################################################################
def cosine_distance_output_shape(shapes):
    return shapes[0]

#################################################################
#                     Inicio da Execucao                        #
#################################################################
DATA_DIR = os.environ["DATA_DIR"]
VQA_DIR = os.path.join(DATA_DIR,"vqa")
IMAGE_DIR = os.path.join(VQA_DIR,"mscoco")
TRIPLES_FILE = os.path.join(DATA_DIR, "triples_train.csv") 

logger.debug("DATA_DIR %s", DATA_DIR)
logger.debug("IMAGE_DIR %s", IMAGE_DIR)
logger.debug("TRIPLES_FILE %s", TRIPLES_FILE)

triples_data = carregar_triplas(TRIPLES_FILE)

logger.debug( "Numero de pares : %d",  len(triples_data))


image_cache = {}
num_pairs = len(triples_data)

for i, (image_filename_l, image_filename_r, _) in enumerate(triples_data):
    if i % 10000 == 0:
        logger.info("images from {:d}/{:d} pairs loaded to cache".format(i, num_pairs))
    if image_filename_l not in image_cache:
        load_image_cache(image_cache, image_filename_l)
    if image_filename_r not in image_cache:
        load_image_cache(image_cache, image_filename_r)
logger.info("images from {:d}/{:d} pairs loaded to cache, COMPLETE".format(i, num_pairs))


datagen_args = dict(rotation_range=10,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    zoom_range=0.2)
datagens = [ImageDataGenerator(**datagen_args),
            ImageDataGenerator(**datagen_args)]
pair_gen = pair_generator(triples_data, image_cache, datagens, 32)
[X1, X2], Y = pair_gen.__next__()

vecs = [np.random.random((10,)), np.random.random((10,))]
s = cosine_distance(vecs)

inception_1 = inception_v3.InceptionV3(weights="imagenet", include_top=True)
inception_2 = inception_v3.InceptionV3(weights="imagenet", include_top=True)

# travando os pesos das camasdas e dando  a cada camada um nome unico
# uma vez que as redes serao combinada um unica rede siamesa
for layer in inception_1.layers:
    layer.trainable = False
    layer.name = layer.name + "_1"
for layer in inception_2.layers:
    layer.trainable = False
    layer.name = layer.name + "_2"

# as saidas da rede serao conecatadas ao ultimo pedaco da rede
vector_1 = inception_1.get_layer("avg_pool_1").output
vector_2 = inception_2.get_layer("avg_pool_2").output

# carregando a rede pre-treinada
siamese_head = load_model(os.path.join(DATA_DIR, "models", "inceptionv3-dot-best.h5"))
for layer in siamese_head.layers:
    print(layer.name, layer.input_shape, layer.output_shape)

predicao = siamese_head([vector_1, vector_2])

model = Model(inputs=[inception_1.input, inception_2.input], outputs=predicao)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#################################################################
#                     Treinamento da Rede                       #
#################################################################

# Parametrizacao da Rede
BATCH_SIZE = 32
NUM_EPOCHS = 10
BEST_MODEL_FILE = os.path.join(DATA_DIR, "models", "inception-ft-best.h5")
FINAL_MODEL_FILE = os.path.join(DATA_DIR, "models", "inception-ft-final.h5")

# Quebrando o dataset em partes para o treinamento
triples_data_trainval, triples_data_test = train_test_split(triples_data, train_size=0.8)
triples_data_train, triples_data_val = train_test_split(triples_data_trainval, train_size=0.9)
