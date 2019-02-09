import os, sys, logging, multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imresize
from keras.applications import mobilenet
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils.generic_utils import CustomObjectScope
from keras.applications.mobilenet import relu6, DepthwiseConv2D

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='../logs/mobilenet_predict_curated.log',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

DATA_DIR = DATA_DIR = os.environ["DATA_DIR"]
VQA_DIR = os.path.join(DATA_DIR, "vqa", "mscoco")
IMAGENET_DIR = os.path.join(DATA_DIR, "ILSVRC", "Data", "DET", "train", "ILSVRC2013_train")

FINAL_MODEL_FILE = os.path.join(DATA_DIR, "vqa", "models", "mobilenet-distilation","mobilenet-destilation-1-dot-best.h5")

PAIRS_DATA_FILE = os.path.join(DATA_DIR, "pairs_data.csv")

logger.debug("DATA_DIR %s", DATA_DIR)
logger.debug("VQA_DIR %s", VQA_DIR)
logger.debug("IMAGENET_DIR %s", IMAGENET_DIR)
logger.debug("FINAL_MODEL_FILE %s", FINAL_MODEL_FILE)


################################################################
def pair_generator(triples, image_cache, datagens, batch_size=32):    
    while True:
        # shuffle once per batch
        #indices = np.random.permutation(np.arange(len(triples)))        
        indices = np.array( range(0, len(triples)-1))
        num_batches = len(triples) // batch_size        
        for bid in range(num_batches):            
            batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]            
            batch = [triples[i] for i in batch_indices]
            X1 = np.zeros((batch_size, 224, 224, 3))
            X2 = np.zeros((batch_size, 224, 224, 3))
            
            for i, (image_filename_l, image_filename_r) in enumerate(batch):
                X1[i] = image_cache[image_filename_l]
                X2[i] = image_cache[image_filename_r]                

            yield [X1, X2]
#################################################################
def load_image_cache(image_cache, image_filename, directory):
    try:
        image = Image.open(os.path.join(directory, image_filename)).convert("RGB")
        image = imresize(image, (224, 224))
        image = image.astype("float32")
        image = mobilenet.preprocess_input(image)
        image_cache[image_filename] = image
    except Exception as e:
        logger.warn("Falha ao ler o arquivo [%s]", os.path.join(directory, image_filename))
        logger.error(e)
#################################################################


with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
    logger.info("Carregando o modelo")
    model = load_model(FINAL_MODEL_FILE)
    logger.info("Modelo carregado com sucesso")

BATCH_SIZE = 64
chunksize = BATCH_SIZE * (10 ** 3)
for data in pd.read_csv(PAIRS_DATA_FILE, names=['filename', 'imagenet_img_id'], chunksize=chunksize):

    image_cache = {}
    pairs_data = data.values.tolist()
    
    logger.info("Carregando cache de imagens")
    for i, (image_filename_l, image_filename_r) in enumerate(pairs_data):        
        if image_filename_l not in image_cache:
            load_image_cache(image_cache, image_filename_l, VQA_DIR)
        if image_filename_r not in image_cache:        
            load_image_cache(image_cache, image_filename_r, IMAGENET_DIR) 
    
    STEPS = len(pairs_data) // BATCH_SIZE

    logger.info("Predizendo similaridades...")        
    data_generator = pair_generator(pairs_data, image_cache, None, BATCH_SIZE)
    n_workers = multiprocessing.cpu_count() - 1
    predicoes = model.predict_generator(data_generator, workers=n_workers, verbose=1, steps=STEPS, use_multiprocessing=False)

    similarities = []
    i = 0      
    for y in predicoes:                    
        pairs_data[i].extend([y[1]])
        similarities.append( pairs_data[i] )          
        i = i + 1 

    filename = os.path.join(DATA_DIR, "predicoes", "model1", "predicoes.csv")       
    logger.info("Salvando arquivo em %s", filename)     
    df = pd.DataFrame(similarities, columns=["mscoco", "imagenet", "similarity"])
    df.to_csv(os.path.join(DATA_DIR, "predicoes", "model1", "predicoes.csv"), mode='a', header=0, index = 0, encoding="utf-8" )

logger.info("Finalizado")