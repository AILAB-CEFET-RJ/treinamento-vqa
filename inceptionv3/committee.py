import os, sys, time, logging, argparse
import pandas as pd
import numpy as np


#################################################################
#               Configurando logs de execucao                   #
#################################################################
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/commitee.log',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

#################################################################
#               Configurando argumentos de entrada              #
#################################################################
parser = argparse.ArgumentParser(description='Commitee for choice image most similar')

parser.add_argument('--file', action = 'store', dest = 'filename',
                           required = True, help = 'Filename of triples file')

parser.add_argument('--method', action = 'store', dest = 'method',
                           required = False, default='max', 
                           help = 'Filename of triples file')


arguments = parser.parse_args()

if(arguments.filename == None or arguments.filename == ''):
    raise Exception('Invalid input filename')
if(not arguments.method in {'mean', 'sum', 'max'} ):
    raise Exception('Invalid method')

#################################################################
#                    Variaveis de execucao                      #
#################################################################
DATA_DIR = os.environ["DATA_DIR"]
PREDICTS_FOLDER = os.path.join(DATA_DIR, "predicoes")
PREDICTS_FILENAME = os.path.join(PREDICTS_FOLDER, arguments.filename)

logger.debug("DATA_DIR %s", DATA_DIR)
logger.debug("PREDICTS_FOLDER %s", PREDICTS_FOLDER)
logger.debug("PREDICTS_FILENAME %s", PREDICTS_FILENAME)
logger.debug("METHOD => [%s]", arguments.method)


predicts = pd.read_csv(PREDICTS_FILENAME)


logger.info("Finalizado")