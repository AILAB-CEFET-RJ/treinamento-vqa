import os
import logging
from keras.models import Model, load_model
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

DATA_DIR = os.environ["DATA_DIR"]
FINAL_MODEL_FILE = os.path.join(DATA_DIR, "models", "inception-ft-final-500.h5")

logger.debug("DATA_DIR %s", DATA_DIR)
logger.debug("FINAL_MODEL_FILE %s", FINAL_MODEL_FILE)


final_model = load_model(FINAL_MODEL_FILE)

Ytest_ = model.predict()

logger.info("Finalizado")
