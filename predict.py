import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
def carregar_pares(vqa_dir, imagenet_dir):
    pairs = []
    for vqa_file in os.listdir(vqa_dir):
         for img_file in os.listdir(imagenet_dir):
             pairs.append( [vqa_file, img_file] )
    return pairs
#################################################################
def load_image_cache(image_cache, image_filename, directory):
    try:
        image = plt.imread(os.path.join(directory, image_filename))
        image = imresize(image, (299, 299))
        image = image.astype("float32")
        image = inception_v3.preprocess_input(image)
        image_cache[image_filename] = image
    except:
        logger.error("Falha ao ler o arquivo [%s]", os.path.join(directory, image_filename))
################################################################
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
            
            for i, (image_filename_l, image_filename_r) in enumerate(batch):                
                if datagens is None or len(datagens) == 0:
                    X1[i] = image_cache[image_filename_l]
                    X2[i] = image_cache[image_filename_r]
                else:
                    X1[i] = datagens[0].random_transform(image_cache[image_filename_l])
                    X2[i] = datagens[1].random_transform(image_cache[image_filename_r])
            yield [X1, X2]
################################################################
def predizer(model):    
    
    ytest, ytest_ = [], []    
    test_pair_gen = pair_generator(pairs_data, image_cache, None, BATCH_SIZE)    
    num_test_steps = len(pairs_data) // BATCH_SIZE
    curr_test_steps = 0
    
    logger.debug( "NUM STEPS PER BATCH : %d",  num_test_steps)

    for [X1test, X2test] in test_pair_gen:
        if curr_test_steps > num_test_steps:
            break        
        Ytest_ = model.predict([X1test, X2test])        
        ytest_.extend(np.argmax(Ytest_, axis=1).tolist())
        curr_test_steps += 1
        if(curr_test_steps % 1000 == 0):
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
IMAGENET_DIR = os.path.join(IMAGE_DIR,"imagenet", "convertido")
VQA_DIR = os.path.join(IMAGE_DIR, "vqa", "teste")

logger.debug("DATA_DIR %s", DATA_DIR)
logger.debug("FINAL_MODEL_FILE %s", FINAL_MODEL_FILE)
logger.debug("TRIPLES_FILE %s", TRIPLES_FILE)
logger.debug("IMAGE_DIR %s", IMAGE_DIR)

logger.debug( "Carregando pares de imagens...")

pairs_data = carregar_pares(VQA_DIR, IMAGENET_DIR)
num_pairs = len(pairs_data)

logger.debug( "Numero de pares : %d",  num_pairs)

image_cache = {}

logger.debug( "carregando imagens")

for i, (image_filename_l, image_filename_r) in enumerate(pairs_data):    
    if image_filename_l not in image_cache:
        load_image_cache(image_cache, image_filename_l, VQA_DIR)
    if image_filename_r not in image_cache:
        load_image_cache(image_cache, image_filename_r, IMAGENET_DIR)

logger.info("imagens carregadas")

test_pair_gen = pair_generator(pairs_data, image_cache, None, None)

logger.info("Carregando o modelo")
model = load_model(FINAL_MODEL_FILE)
logger.info("Modelo carregado com sucesso")

BATCH_SIZE = 5

#acc,cm, y = predizer(model)
logger.info("Predizendo similaridades")
predicoes = predizer(model)
logger.info("Pronto")

logger.info("Salvando as predicoes")

tam = len(predicoes)

for i in range(0, tam-1):    
    pairs_data[i].append(predicoes[i])


df = pd.DataFrame(pairs_data, columns=["mscoco", "imagenet", "similar"])
df.to_csv(os.path.join(DATA_DIR, "predicoes.csv"), header=0, index = 0, compression="gzip")
logger.info("salvo em %s", os.path.join(DATA_DIR, "predicoes.csv"))

logger.info("Finalizado")
