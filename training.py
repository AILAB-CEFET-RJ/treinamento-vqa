import os

from keras import backend as K
from keras.applications import inception_v3
from keras.callbacks import ModelCheckpoint, CSVLogger
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

def evaluate_model(model):
    ytest, ytest_ = [], []
    test_pair_gen = pair_generator(triples_data_test, image_cache, None, BATCH_SIZE)
    num_test_steps = len(triples_data_test) // BATCH_SIZE
    curr_test_steps = 0
    for [X1test, X2test], Ytest in test_pair_gen:
        if curr_test_steps > num_test_steps:
            break
        Ytest_ = model.predict([X1test, X2test])
        ytest.extend(np.argmax(Ytest, axis=1).tolist())
        ytest_.extend(np.argmax(Ytest_, axis=1).tolist())
        curr_test_steps += 1
    acc = accuracy_score(ytest, ytest_)
    cm = confusion_matrix(ytest, ytest_)
    return acc, cm

#################################################################
#                     Inicio da Execucao                        #
#################################################################
DATA_DIR = os.environ["DATA_DIR"]
VQA_DIR = os.path.join(DATA_DIR,"vqa")
IMAGE_DIR = os.path.join(VQA_DIR,"mscoco")
TRIPLES_FILE = os.path.join(DATA_DIR, "triples_train_500.csv") 

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

logger.debug("compilando o modelo")
model = Model(inputs=[inception_1.input, inception_2.input], outputs=predicao)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#################################################################
#                     Treinamento da Rede                       #
#################################################################
logger.info(" ####### Inicio do treinamento ######")

# Parametrizacao da Rede
BATCH_SIZE = 32
NUM_EPOCHS = 10
BEST_MODEL_FILE = os.path.join(DATA_DIR, "models", "inception-ft-best.h5")
FINAL_MODEL_FILE = os.path.join(DATA_DIR, "models", "inception-ft-final-500.h5")

logger.info("TAMANHO BATCH %s", BATCH_SIZE)
logger.info("NUM DE EPOCAS %s", NUM_EPOCHS)
logger.info("MELHOR MODELO %s", BEST_MODEL_FILE)
logger.info("MODELO FINAL %s", FINAL_MODEL_FILE)

# Quebrando o dataset em partes para o treinamento
logger.info("Dividindo o dataset")
triples_data_trainval, triples_data_test = train_test_split(triples_data, train_size=0.8)
triples_data_train, triples_data_val = train_test_split(triples_data_trainval, train_size=0.9)

logger.debug("DATA TRAIN %d", len(triples_data_trainval))
logger.debug("DATA TEST %d", len(triples_data_test))
logger.debug("DATA VAL TRAIN %d", len(triples_data_train))
logger.debug("DATA VAL TEST %d", len(triples_data_val))

logger.info("Entrada: Concatenando Vetores")
datagen_args = dict(rotation_range=10,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    zoom_range=0.2)
datagens = [ImageDataGenerator(**datagen_args),
            ImageDataGenerator(**datagen_args)]
train_pair_gen = pair_generator(triples_data_train, image_cache, datagens, BATCH_SIZE)
val_pair_gen = pair_generator(triples_data_val, image_cache, None, BATCH_SIZE)
logger.info("pronto")

num_train_steps = len(triples_data_train) // BATCH_SIZE
num_val_steps = len(triples_data_val) // BATCH_SIZE
logger.debug("passos de treinamento por epoca %d", num_train_steps)
logger.debug("passos de validacao por epoca %d",  num_val_steps)


logger.info("Ajustando o modelo")
csv_logger = CSVLogger(os.path.join("logs", 'training_epochs.csv'))
checkpoint = ModelCheckpoint(filepath=BEST_MODEL_FILE, save_best_only=True)

logger.debug("gravando dados das epocas em %s", os.path.join("logs", 'training_epochs.csv'))
callback_list = [csv_logger, checkpoint]

history = model.fit_generator(train_pair_gen, 
                             steps_per_epoch=num_train_steps,
                             epochs=NUM_EPOCHS,
                             validation_data=val_pair_gen,
                             validation_steps=num_val_steps,
                             callbacks=callback_list)

logger.info("gerando os graficos de treinamento")
plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["acc"], color="blue", label="train")
plt.plot(history.history["val_acc"], color="red", label="validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"], color="blue", label="train")
plt.plot(history.history["val_loss"], color="red", label="validation")
plt.legend(loc="best")

plt.tight_layout()
plt.savefig("graphs/best.png")
plt.close()

logger.info("salvando o modelo final em %s", FINAL_MODEL_FILE)
model.save(FINAL_MODEL_FILE, overwrite=True)


logger.info("==== Avaliando Resultados: Modelo final sobre o conjunto de dados ====")
final_model = load_model(FINAL_MODEL_FILE)
acc, cm = evaluate_model(final_model)
logger.info("Precisao: {:.3f}".format(acc))
logger.info("Matriz de Confusao")
logger.info(cm)

logger.info("==== Avaliando Resultados: Melhor modelo sobre o conjunto de dados ====")
best_model = load_model(BEST_MODEL_FILE)
acc, cm = evaluate_model(best_model)
logger.info("Precisao: {:.3f}".format(acc))
logger.info("Matriz de Confusao")
logger.info(cm)

logger.info("Fim da execucao")