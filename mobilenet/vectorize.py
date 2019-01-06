#################################################################
#             Script para gerar vetores das imagens             #
#################################################################

import os, math, sys
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from scipy.misc import imresize
from keras.applications import mobilenet, xception

#################################################################
#               Gerando lotes para treinamento                  #
#################################################################
def image_batch_generator(image_names, batch_size):        
    num_batches = len(image_names) // batch_size        
    for i in range(num_batches):
        batch = image_names[i * batch_size : (i + 1) * batch_size]
        yield batch
    batch = image_names[(i+1) * batch_size:]
    yield batch


def vectorize_images(image_dir, image_size, preprocessor, 
                     model, vector_file, batch_size=32):
    
    if( os.path.isfile(vector_file) ):
        print("Removendo arquivo anterior", vector_file)
        os.remove(vector_file)                
    
    image_names = os.listdir(image_dir)
    print(len(image_names), "imagens encontradas")
    num_vecs =  0
    
    fvec = open(vector_file, "w")
    for image_batch in image_batch_generator(image_names, batch_size):
        batched_images = []
        for image_name in image_batch:
            image = plt.imread(os.path.join(image_dir, image_name))
            image = imresize(image, (image_size, image_size))
            batched_images.append(image)
        X = preprocessor(np.array(batched_images, dtype="float32"))        
        vectors = model.predict(X)        
        for i in range(vectors.shape[0]):
            if num_vecs % 1000 == 0:
                print("{:d} vectors generated".format(num_vecs))             
            image_vector = ",".join(["{:.5e}".format(v) for v in vectors[i].tolist()])
            fvec.write("{:s}\t{:s}\n".format(image_batch[i], image_vector))
            num_vecs += 1
    print("{:d} vectors generated".format(num_vecs))
    fvec.close()


#################################################################
#                          Constantes                           #
#################################################################
IMAGE_SIZE = 224
DATA_DIR = os.environ["DATA_DIR"]
VQA_DIR = os.path.join(DATA_DIR,"vqa")
IMAGE_DIR = os.path.join(VQA_DIR,"convertidas")
VECTOR_FILE = os.path.join(VQA_DIR, "vectors", "mobilenet-vectors-destilation.tsv")

print("DATA_DIR", DATA_DIR)
print("VQA_DIR ", VQA_DIR)
print("IMAGE_DIR ", IMAGE_DIR)
print("VECTOR_FILE", VECTOR_FILE)


#################################################################
#                       Inicio da Execucao                      #
#################################################################
mobile_model = mobilenet.MobileNet(weights="imagenet", include_top=True, depth_multiplier=1, dropout=1e-3)

model = Model(inputs=mobile_model.input,
             outputs=mobile_model.get_layer("global_average_pooling2d_1").output)
preprocessor = mobilenet.preprocess_input


vectorize_images(IMAGE_DIR, IMAGE_SIZE, preprocessor, model, VECTOR_FILE)

print(VECTOR_FILE, "criado com sucesso !!!")
