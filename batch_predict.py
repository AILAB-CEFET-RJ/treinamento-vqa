import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from util.ts_iterator import threadsafe_iter

#################################################################
#               Configurando logs de execucao                   #
#################################################################
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/batch_predict.log',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

#################################################################
def tic():
    global _start_time 
    _start_time = time.time()

def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    logger.info('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))
#################################################################

def threadsafe_decorator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

#################################################################
def gerar_pares_imagens(vqa_file, imagenet_filenames):
    image_pairs = []
    for imagenet_file in imagenet_filenames:
        image_pairs.append([vqa_file, imagenet_file])    
    return image_pairs

#################################################################
def load_image_cache(image_cache, image_filename, directory):
    try:
        image = Image.open(os.path.join(directory, image_filename)).convert("RGB")
        image = imresize(image, (299, 299))
        image = image.astype("float32")
        image = inception_v3.preprocess_input(image)
        image_cache[image_filename] = image
    except Exception as e:
        logger.warn("Falha ao ler o arquivo [%s]", os.path.join(directory, image_filename))
        logger.error(e)       
################################################################
@threadsafe_decorator
def pair_generator(triples, image_cache, datagens, batch_size=32):    
    while True:
        # shuffle once per batch
        #indices = np.random.permutation(np.arange(len(triples)))        
        indices = np.array( range(0, len(triples)-1))
        num_batches = len(triples) // batch_size        
        for bid in range(num_batches):            
            batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]            
            batch = [triples[i] for i in batch_indices]
            X1 = np.zeros((batch_size, 299, 299, 3))
            X2 = np.zeros((batch_size, 299, 299, 3))
            
            for i, (image_filename_l, image_filename_r) in enumerate(batch):                
                if datagens is None or len(datagens) == 0:                   
                    X1[i] = image_cache[image_filename_l]
                    X2[i] = image_cache[image_filename_r]                    
                else:
                    X1[i] = datagens[0].random_transform(image_cache[image_filename_l])
                    X2[i] = datagens[1].random_transform(image_cache[image_filename_r])
            yield [X1, X2]

################################################################
def obter_nome_arquivos_imagenet(synset_list):
    imagenet_data = []
    for synset in synsets:        
        logger.info("carregando o synset [%s]", synset[0])
        synset_path = os.path.join(IMAGENET_DIR, synset[0])
        
        if os.path.isdir(synset_path):
            for imagenet_file in os.listdir(synset_path):
                imagenet_data.append(os.path.join(synset[0], imagenet_file))
                load_image_cache(image_cache, os.path.join(synset[0], imagenet_file), IMAGENET_DIR)  
        else:
            logger.warn("%s nao existe", synset_path) 
    return imagenet_data        

################################################################
def load_synset_list(path):
    df = pd.read_csv(path, encoding="utf-8", sep=' ', usecols=[0])
    return df.values
################################################################
def load_vqa_filenames_list(path):
    df = pd.read_csv(path, names=["filename"], encoding="utf-8", header=1)
    return df.values
################################################################
DATA_DIR = os.environ["DATA_DIR"]
#FINAL_MODEL_FILE = os.path.join(DATA_DIR, "models", "inception-ft-best.h5")
FINAL_MODEL_FILE = os.path.join(DATA_DIR, "vqa", "models", "distilation","inception-training-distlation3-ft-best.h5")
TRIPLES_FILE = os.path.join(DATA_DIR, "triplas_imagenet_vqa.csv") 
IMAGE_DIR = DATA_DIR
IMAGENET_DIR = os.path.join(IMAGE_DIR, "ILSVRC", "Data", "DET", "train", "ILSVRC2013_train")
#IMAGENET_DIR =  os.path.join(IMAGE_DIR, "ILSVRC2013_train")
VQA_DIR = os.path.join(IMAGE_DIR, "vqa", "mscoco")
BATCH_SIZE = 192

logger.debug("DATA_DIR %s", DATA_DIR)
logger.debug("FINAL_MODEL_FILE %s", FINAL_MODEL_FILE)
logger.debug("TRIPLES_FILE %s", TRIPLES_FILE)
logger.debug("IMAGE_DIR %s", IMAGE_DIR)

logger.debug("IMAGENET_DIR %s", IMAGENET_DIR)
logger.debug("VQA_DIR %s", VQA_DIR)

logger.info("Carregando o modelo")
model = load_model(FINAL_MODEL_FILE)
logger.info("Modelo carregado com sucesso")

logger.debug( "Carregando pares de imagens...")

synsets = load_synset_list(os.path.join(DATA_DIR, "LOC_synset_mapping.txt"))

# Apenas os N synsets 
OFFSET_SYNSET = 0
NB_SYNSETS = 10

synsets = synsets[OFFSET_SYNSET:NB_SYNSETS]
vqa_filenames_list = load_vqa_filenames_list(os.path.join(DATA_DIR, "mscoco_cats.csv"))

tic()
image_cache = {}
imagenet_filename = obter_nome_arquivos_imagenet(synsets)
tac()

#for vqa_file in os.listdir(VQA_DIR):
for filename in vqa_filenames_list:  
    vqa_file = filename[0]
    vqa_image_path = os.path.join(VQA_DIR,vqa_file)

     if os.path.exists("predicoes_{:s}.csv".format(vqa_file)):
         continue

    load_image_cache(image_cache, vqa_file, VQA_DIR)

    logger.info("processando a imagem [%s]", vqa_image_path)
    similarities = []
    
    tic()  # Marca o tempo de inicio da execucao
    
    pairs_data = gerar_pares_imagens(vqa_file, imagenet_filename)
    num_pairs = len(pairs_data)
    logger.debug( "Numero de pares : %d",  num_pairs)
    
    STEPS = num_pairs // BATCH_SIZE
    
    generator=pair_generator(pairs_data, image_cache, None, BATCH_SIZE)

    logger.info("Predizendo similaridades...")        
    predicoes = model.predict_generator(generator, verbose=1, steps=STEPS, max_queue_size=10, workers=3, use_multiprocessing=False)
    logger.info("pronto")

    logger.debug("gerando dados de predicao")
    
    i = 0      
    for y in predicoes:            
        #pairs_data[i].extend([y[1]])
        #similarities.append( pairs_data[i] )
        _,imagenet_name = os.path.split(pairs_data[i][1])
        similarities.append([imagenet_name, y[1]])
        i = i + 1
    logger.debug("pronto")
    
    logger.info("Salvando as predicoes...")
    predict_filename = "{:s}.csv".format(vqa_file)        
    
    #df = pd.DataFrame(similarities, columns=["mscoco", "imagenet", "similarity"])
    df = pd.DataFrame(similarities, columns=["imagenet", "similarity"])
    df.to_csv(os.path.join(DATA_DIR, "predicoes", predict_filename), mode='a', header=0, index = 0, encoding="utf-8" )
    logger.info("salvo em %s", os.path.join(DATA_DIR, "predicoes" , predict_filename))
    del image_cache[vqa_file]
    tac() # Marca o tempo de fim da execucao

logger.info("Finalizado !!!")
