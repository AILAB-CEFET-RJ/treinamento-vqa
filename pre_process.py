from keras import backend as K
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers.core import Activation, Dense, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.models import Model, load_model
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use('Agg')

import logging

#################################################################
#               Configurando logs de execucao                   #
#################################################################
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/pre_process.log',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

#################################################################
#                           Métodos                             #
#################################################################
def carregar_triplas(lista_triplas):
    image_triples = pd.read_csv(lista_triplas, sep=",", header=0, names=["left","right","similar"])
    return image_triples.values

def carregar_vetores(vector_file):
    vec_dict = {}
    fvec = open(vector_file, "r")
    for line in fvec:
        image_name, image_vec = line.strip().split("\t")
        vec = np.array([float(v) for v in image_vec.split(",")])
        vec_dict[image_name] = vec
    fvec.close()
    return vec_dict

def train_test_split(triples, splits):
    assert sum(splits) == 1.0
    split_pts = np.cumsum(np.array([0.] + splits))
    indices = np.random.permutation(np.arange(len(triples)))
    shuffled_triples = [triples[i] for i in indices]
    data_splits = []
    for sid in range(len(splits)):
        start = int(split_pts[sid] * len(triples))
        end = int(split_pts[sid + 1] * len(triples))
        data_splits.append(shuffled_triples[start:end])
    return data_splits

def data_generator(triples, vec_size, vec_dict, batch_size=32):
    while True:
        # shuffle once per batch
        indices = np.random.permutation(np.arange(len(triples)))
        num_batches = len(triples) // batch_size
        for bid in range(num_batches):
            batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]
            batch = [triples[i] for i in batch_indices]
            yield batch_to_vectors(batch, vec_size, vec_dict)

def batch_to_vectors(batch, vec_size, vec_dict):
    X1 = np.zeros((len(batch), vec_size))
    X2 = np.zeros((len(batch), vec_size))
    Y = np.zeros((len(batch), 2))
    for tid in range(len(batch)):
        X1[tid] = vec_dict[batch[tid][0]]
        X2[tid] = vec_dict[batch[tid][1]]
        Y[tid] = [1, 0] if batch[tid][2] == 0 else [0, 1]
    return ([X1, X2], Y)

def get_model_file(data_dir, vector_name, merge_mode, borf):
    return os.path.join(data_dir, "models", "{:s}-{:s}-{:s}.h5"
                        .format(vector_name, merge_mode, borf))

def evaluate_model(model_file, test_gen):
    model_name = os.path.basename(model_file)
    model = load_model(model_file)
    logger.debug("=== Evaluating model: {:s} ===".format(model_name))
    ytrue, ypred = [], []
    num_test_steps = len(test_triples) // BATCH_SIZE
    for i in range(num_test_steps):
        (X1, X2), Y = test_gen.__next__()
        Y_ = model.predict([X1, X2])
        ytrue.extend(np.argmax(Y, axis=1).tolist())
        ypred.extend(np.argmax(Y_, axis=1).tolist())
    accuracy = accuracy_score(ytrue, ypred)
    logger.info("\nAccuracy: {:.3f}".format(accuracy))
    logger.info("\nConfusion Matrix")
    logger.info(confusion_matrix(ytrue, ypred))
    logger.info("\nClassification Report")
    logger.info(classification_report(ytrue, ypred))
    return accuracy
#################################################################
#                     Inicio da Execucao                        #
#################################################################
DATA_DIR = os.environ["DATA_DIR"]
VQA_DIR = os.path.join(DATA_DIR, "vqa")
IMAGE_DIR = os.path.join(VQA_DIR, "mscoco")
TRIPLES_FILE = os.path.join(DATA_DIR, "triples_train.csv") 
#VALIDATION_TRIPLES_FILE = os.path.join(DATA_DIR, "triples_val.csv") 

logger.debug("DATA_DIR %s", DATA_DIR)
logger.debug("IMAGE_DIR %s", IMAGE_DIR)
logger.debug("TRIPLES_FILE %s", TRIPLES_FILE)
#logger.debug("VAL_TRIPLES_FILE %s", VALIDATION_TRIPLES_FILE)

BATCH_SIZE = 32
NUM_EPOCHS = 10

VECTORIZERS = ["InceptionV3"]
MERGE_MODES = ["Dot"]
scores = np.zeros((len(VECTORIZERS), len(MERGE_MODES)))

logger.info("carregando triplas")
image_triples = carregar_triplas(TRIPLES_FILE)
#validation_triples = carregar_triplas(VALIDATION_TRIPLES_FILE)
logger.info("Pronto")

logger.info("Dividindo o dataset")
#train_triples = image_triples
#val_triples, test_triples = train_test_split(validation_triples, splits=[0.5, 0.5])

train_triples, val_triples, test_triples = train_test_split(image_triples, splits=[0.7, 0.1, 0.2])

logger.debug("Train size %d", len(train_triples))
logger.debug("Validation size %d", len(val_triples))
logger.debug("Test size %d", len(test_triples))

VECTOR_SIZE = 2048
VECTOR_FILE = os.path.join(DATA_DIR, "inception-vectors.tsv")
#VALIDATION_VECTOR_FILE = os.path.join(DATA_DIR, "inception-vectors-validation.tsv")
logger.info("VECTOR_FILE %s", VECTOR_FILE)

logger.info("Carregando vetores")
vec_dict = carregar_vetores(VECTOR_FILE)
#validation_dict = carregar_vetores(VALIDATION_VECTOR_FILE)
logger.info("pronto")

logger.info("Entrada: Concatenando Vetores")
train_gen = data_generator(train_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
val_gen = data_generator(val_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
logger.info("pronto")

logger.info(" ####### Inicio do treinamento ######")

input_1 = Input(shape=(VECTOR_SIZE,))
input_2 = Input(shape=(VECTOR_SIZE,))
merged = Concatenate(axis=-1)([input_1, input_2])

fc1 = Dense(512, kernel_initializer="glorot_uniform")(merged)
fc1 = Dropout(0.2)(fc1)
fc1 = Activation("relu")(fc1)

fc2 = Dense(128, kernel_initializer="glorot_uniform")(fc1)
fc2 = Dropout(0.2)(fc2)
fc2 = Activation("relu")(fc2)

pred = Dense(2, kernel_initializer="glorot_uniform")(fc2)
pred = Activation("softmax")(pred)

model = Model(inputs=[input_1, input_2], outputs=pred)

logger.info("compilando o modelo")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
logger.info("pronto")

best_model_name = get_model_file(DATA_DIR, "inceptionv3", "dot", "best")
logger.debug("model name : %s", best_model_name)

csv_logger = CSVLogger(os.path.join("logs", 'training_epochs.csv'))
checkpoint = ModelCheckpoint(best_model_name, save_best_only=True)

logger.debug("gravando dados das epocas em %s", os.path.join("logs", 'training_epochs.csv'))
callback_list = [csv_logger, checkpoint]

train_steps_per_epoch = len(train_triples) // BATCH_SIZE
val_steps_per_epoch = len(val_triples) // BATCH_SIZE

logger.debug("passos de treinamento por epoca %d", len(train_triples) // BATCH_SIZE)
logger.debug("passos de valicao por epoca %d",  len(val_triples) // BATCH_SIZE)

history = model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, 
                              epochs=NUM_EPOCHS, 
                              validation_data=val_gen, validation_steps=val_steps_per_epoch,
                              callbacks=callback_list)

logger.info("gravando graficos de treinamento")

plt.subplot(211)
plt.title("Loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="b", label="validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Accuracy")
plt.plot(history.history["acc"], color="r", label="train")
plt.plot(history.history["val_acc"], color="b", label="validation")
plt.legend(loc="best")

plt.savefig("graphs/best.png")
plt.close()

final_model_name = get_model_file(DATA_DIR, "inceptionv3", "dot", "final")
logger.info("salvando o modelo em %s", final_model_name)

model.save(final_model_name)
test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
final_accuracy = evaluate_model(final_model_name, test_gen)
logger.debug("final accuracy %s", final_accuracy)

test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
best_accuracy = evaluate_model(best_model_name, test_gen)
logger.debug("best accuracy %s", best_accuracy)

scores[0, 0] = best_accuracy if best_accuracy > final_accuracy else final_accuracy

logger.info("SCORE : %s", scores[0,0])

logger.info("Fim da execucao")
