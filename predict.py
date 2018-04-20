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

#################################################################
#               Configurando logs de execucao                   #
#################################################################
def carregar_pares(vqa_file, synset_dir):
    image_pairs = []    
    for imagenet_file in os.listdir(os.path.join(IMAGENET_DIR, synset_dir)):
        image_pairs.append( [vqa_file, os.path.join(synset_dir, imagenet_file)])
    return image_pairs

#################################################################
def load_image_cache(image_cache, image_filename, directory):
    try:
        image = Image.open(os.path.join(directory, image_filename)).convert("RGB")
        image = imresize(image, (299, 299))
        image = image.astype("float32")
        image = inception_v3.preprocess_input(image)
        image_cache[image_filename] = image
    except:
        logger.warn("Falha ao ler o arquivo [%s]", os.path.join(directory, image_filename))        
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
            X1 = np.zeros((batch_size, 299, 299, 3))
            X2 = np.zeros((batch_size, 299, 299, 3))
            X3, X4 = [],[]
            
            for i, (image_filename_l, image_filename_r) in enumerate(batch):                
                X3.append(image_filename_l)
                X4.append(image_filename_r)
                if datagens is None or len(datagens) == 0:                   
                    X1[i] = image_cache[image_filename_l]
                    X2[i] = image_cache[image_filename_r]                    
                else:
                    X1[i] = datagens[0].random_transform(image_cache[image_filename_l])
                    X2[i] = datagens[1].random_transform(image_cache[image_filename_r])
            yield [X1, X2, X3, X4]
################################################################
def predizer(model):    
    
    ytest_ = []
    test_pair_gen = pair_generator(pairs_data, image_cache, None, BATCH_SIZE)    
    num_test_steps = len(pairs_data) // BATCH_SIZE
    curr_test_steps = 0
    
    logger.debug( "NUM STEPS PER BATCH : %d",  num_test_steps)
    logger.debug( "BATCH SIZE : %d",  BATCH_SIZE)
        
    for [X1test, X2test, image_l, image_r] in test_pair_gen:
        if curr_test_steps > num_test_steps:
            break        
        y = model.predict([X1test, X2test])        
        ytest_.extend(np.argmax(y, axis=1).tolist())
        
        print(image_filename_l, image_filename_r)
        print(ytest_)    
        sys.exit()


        curr_test_steps += 1
        if(curr_test_steps % 100 == 0):
            logger.debug("%s pares analisados", curr_test_steps)        
    #acc = accuracy_score(ytest, ytest_)
    #cm = confusion_matrix(ytest, ytest_)
    #return acc, cm, ytest
    return ytest_
################################################################
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

logger.info("Carregando o modelo")
model = load_model(FINAL_MODEL_FILE)
logger.info("Modelo carregado com sucesso")

logger.debug( "Carregando pares de imagens...")

for vqa_file in os.listdir(VQA_DIR):
    vqa_image_path = os.path.join(VQA_DIR,vqa_file)
    logger.info("processando a imagem [%s]", vqa_image_path)
    similarities = []
    
    for synset_dir in os.listdir(IMAGENET_DIR):
        logger.info("processando o synset [%s]", synset_dir)
        pairs_data = carregar_pares(vqa_file, synset_dir)
        num_pairs = len(pairs_data)

        logger.debug( "Numero de pares : %d",  num_pairs)
        image_cache = {}

        logger.info( "carregando imagens...")
        valid_pairs = []
        
        for i, (image_filename_l, image_filename_r) in enumerate(pairs_data):        
            if image_filename_l not in image_cache:
                load_image_cache(image_cache, image_filename_l, VQA_DIR)
            if image_filename_r not in image_cache:        
                load_image_cache(image_cache, image_filename_r, IMAGENET_DIR)        
            
        test_pair_gen = pair_generator(pairs_data, image_cache, None, None)
        logger.debug( "pronto")
        BATCH_SIZE = 64

        #acc,cm, y = predizer(model)
        logger.info("Predizendo similaridades...")
        predicoes = predizer(model)
        logger.info("pronto")
       
        print(predicoes)
        sys.exit()
    
    logger.info("Salvando as predicoes...")
    predict_filename = "predicoes_{:s}.csv".format(vqa_file)

    df = pd.DataFrame(predicoes, columns=["mscoco", "imagenet", "similar"])
    df.to_csv(os.path.join(DATA_DIR, "predicoes", predict_filename), header=0, index = 0, compression="gzip")
    logger.info("salvo em %s", os.path.join(DATA_DIR, "predicoes" , predict_filename))
    
logger.info("Finalizado !!!")
