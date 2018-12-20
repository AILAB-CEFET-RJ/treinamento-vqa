import os, sys, time, logging


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

def load_images_names(synset_dir):
    image_names = []
    synset_path = os.path.join(IMAGENET_DIR, synset_dir)
    if os.path.isdir(synset_path):
        for imagenet_file in os.listdir(synset_path):
            image_names.append(os.path.join(synset_dir, imagenet_file))
    else:
        logger.debug("%s nao existe", synset_path)        
    return image_names


def load_image_cache(image_cache, image_filename, directory):
    try:
        image = Image.open(os.path.join(directory, image_filename)).convert("RGB")
        image = imresize(image, (299, 299))
        image = image.astype("float32")
        image = inception_v3.preprocess_input(image)
        image_cache[image_filename] = image
    except:
        logger.warn("Falha ao ler o arquivo [%s]", os.path.join(directory, image_filename))        

def load_synset_list(path):
    df = pd.read_csv(path, names=["synset"], encoding="utf-8", header=1)
    return df.values

DATA_DIR = os.environ["DATA_DIR"]
FINAL_MODEL_FILE = os.path.join(DATA_DIR, "models", "inception-ft-best.h5")
TRIPLES_FILE = os.path.join(DATA_DIR, "triplas_imagenet_vqa.csv") 
IMAGE_DIR = DATA_DIR
IMAGENET_DIR = os.path.join(IMAGE_DIR, "ILSVRC", "Data", "DET", "train", "ILSVRC2013_train")
VQA_DIR = os.path.join(IMAGE_DIR, "vqa", "train2014")

logger.debug("DATA_DIR %s", DATA_DIR)
logger.debug("FINAL_MODEL_FILE %s", FINAL_MODEL_FILE)
logger.debug("TRIPLES_FILE %s", TRIPLES_FILE)
logger.debug("IMAGE_DIR %s", IMAGE_DIR)

logger.debug("IMAGENET_DIR %s", IMAGENET_DIR)
logger.debug("VQA_DIR %s", VQA_DIR)


logger.debug( "Carregando pares de imagens...")

synsets = load_synset_list(os.path.join(DATA_DIR, "synsets_dog_cat.csv"))

image_cache = {}

for synset in synsets:
    synset_dir = synset[0]
    logger.info("processando o synset [%s]", synset_dir)
    image_names = load_images_names(synset_dir)    

    for imagenet_image in image_names:
        load_image_cache(image_cache, imagenet_image, IMAGENET_DIR)
    
    print(image_cache)
    sys.exit()