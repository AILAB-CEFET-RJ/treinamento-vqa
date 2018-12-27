import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from keras import backend as K
from keras.applications import inception_v3
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
                    filename='logs/predict_generator.log',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
#################################################################
def pair_generator(triples, image_cache, datagens, batch_size=32):
    while True:        
        indices = np.array( range(0, len(triples)-1))
        num_batches = len(triples) // batch_size
        for bid in range(num_batches):
            batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]
            batch = [triples[i] for i in batch_indices]
            X1 = np.zeros((batch_size, 299, 299, 3))
            X2 = np.zeros((batch_size, 299, 299, 3))
            Y = np.zeros((batch_size, 2))
            for i, (image_filename_l, image_filename_r) in enumerate(batch):
                try:
                    if datagens is None or len(datagens) == 0:
                        X1[i] = image_cache[image_filename_l]
                        X2[i] = image_cache[image_filename_r]
                    else:                                            
                        X1[i] = datagens[0].random_transform(image_cache[image_filename_l])
                        X2[i] = datagens[1].random_transform(image_cache[image_filename_r])                    
                except:
                    logger.error("FALHA AO PROCESSAR L : %s - R : %s", image_filename_l, image_filename_r)
                    continue
            yield [X1, X2]
################################################################            
def carregar_triplas(lista_triplas):
    image_triples = pd.read_csv(lista_triplas, sep=",", header=0, names=["left","right","similar"])
    return image_triples.values
#################################################################
def obter_nome_arquivos_imagenet(synset_list):
    imagenet_data = []
    for synset in synset_list:        
        logger.info("carregando o synset [%s]", synset)
        synset_path = os.path.join(IMAGENET_DIR, synset, "images")
        
        if os.path.isdir(synset_path):
            for imagenet_file in os.listdir(synset_path):
                imagenet_data.append(os.path.join(synset, "images",imagenet_file))                
        else:
            logger.warn("%s nao existe", synset_path) 
    return imagenet_data        
################################################################
def load_vqa_filenames_list(path):
    df = pd.read_csv(path, names=["filename"], encoding="utf-8", header=1)
    return df.values
#################################################################
def load_image_cache(image_cache, image_filename, directory):
    image = plt.imread(os.path.join(directory, image_filename))
    image = imresize(image, (299, 299))
    image = image.astype("float32")
    image = inception_v3.preprocess_input(image)
    image_cache[image_filename] = image
################################################################
def gerar_triplas(vqa_list, imagenet_list):
    triples = []
    for vqa_file in vqa_filenames_list:
        vqa_img = vqa_file[0]
        for imagenet_img in imagenet_list:
            triples.append([vqa_img, imagenet_img])
    return triples
################################################################
def load_synset_list(path):
    df = pd.read_csv(path, encoding="utf-8", sep=' ', usecols=[0])
    return df.values
#################################################################
DATA_DIR = os.environ["DATA_DIR"]
#FINAL_MODEL_FILE = os.path.join(DATA_DIR, "models", "inception-ft-best.h5")
FINAL_MODEL_FILE = os.path.join(DATA_DIR, "vqa", "models", "distilation","inception-training-distlation3-ft-best.h5")
TRIPLES_FILE = os.path.join(DATA_DIR, "triplas_imagenet_vqa.csv") 
IMAGE_DIR = DATA_DIR
IMAGENET_DIR = os.path.join(IMAGE_DIR, "tiny-imagenet-200", "train")
VQA_DIR = os.path.join(IMAGE_DIR, "vqa", "mscoco")
BATCH_SIZE = 128
TRIPLES_FILE = os.path.join(DATA_DIR, "vqa", "distilation", "triples_4.csv") 

logger.debug("DATA_DIR %s", DATA_DIR)
logger.debug("FINAL_MODEL_FILE %s", FINAL_MODEL_FILE)
logger.debug("TRIPLES_FILE %s", TRIPLES_FILE)
logger.debug("IMAGE_DIR %s", IMAGE_DIR)

logger.debug("IMAGENET_DIR %s", IMAGENET_DIR)
logger.debug("VQA_DIR %s", VQA_DIR)
logger.debug("TRIPLES_FILE %s", TRIPLES_FILE)
################################################################
logger.info("Carregando o modelo")
#model = load_model(FINAL_MODEL_FILE)
logger.info("Modelo carregado com sucesso")
################################################################
logger.debug( "Carregando pares de imagens...")

synsets = os.listdir(IMAGENET_DIR)
vqa_filenames_list = load_vqa_filenames_list(os.path.join(DATA_DIR, "mscoco_cats.csv"))


logger.debug("quantidade de synsets %s", len(synsets))

for synset in synsets:
    logger.debug("synset %s", synset)
    imagenet_filenames_list = obter_nome_arquivos_imagenet([synset])

    if len(imagenet_filenames_list) == 0:
        continue

    image_cache = {}
    triples_data = gerar_triplas(vqa_filenames_list, imagenet_filenames_list)

    num_pairs = len(triples_data)
    logger.info("num pares %s", num_pairs)

    ################################################################
    for i, (image_filename_l, image_filename_r) in enumerate(triples_data):
        if i % 10000 == 0:
            logger.info("images from {:d}/{:d} pairs loaded to cache".format(i, num_pairs))
        if image_filename_l not in image_cache:
            load_image_cache(image_cache, image_filename_l, VQA_DIR)
        if image_filename_r not in image_cache:
            load_image_cache(image_cache, image_filename_r, IMAGENET_DIR)
    logger.info("images from {:d}/{:d} pairs loaded to cache, COMPLETE".format(i, num_pairs))
    sys.exit()
    ################################################################
    datagen_args = dict(rotation_range=10,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        zoom_range=0.2)
    datagens = [ImageDataGenerator(**datagen_args),
                ImageDataGenerator(**datagen_args)]
    generator = pair_generator(triples_data, image_cache, datagens, BATCH_SIZE)

    ################################################################

    num_steps = len(triples_data) // BATCH_SIZE
    logger.debug("passos por epoca %d", num_steps)

    logger.info("Predizendo similaridades...")        
    predicoes = model.predict_generator(generator, verbose=1, steps=num_steps, max_queue_size=10, workers=3, use_multiprocessing=False)
    logger.info("pronto")

    ################################################################
    logger.debug("gerando dados de predicao")
    i = 0      
    for y in predicoes:
        _,imagenet_name = os.path.split(pairs_data[i][1])
        similarities.append([imagenet_name, y[1]])
        i = i + 1
    logger.debug("pronto")
    ################################################################
    logger.info("Salvando as predicoes...")
    predict_filename = "{}_predicoes.csv".format(synset) 

    df = pd.DataFrame(similarities, columns=["imagenet", "similarity"])
    df.to_hdf(os.path.join(DATA_DIR, "predicoes", predict_filename), mode='a', key="data" )
    logger.info("salvo em %s", os.path.join(DATA_DIR, "predicoes" , predict_filename))
    ################################################################
logger.info("Finalizado !!!")
