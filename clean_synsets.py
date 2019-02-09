import sys, os
import pandas as pd

DATA_DIR = os.environ["DATA_DIR"]
IMAGE_DIR = DATA_DIR
IMAGENET_DIR = os.path.join(IMAGE_DIR, "ILSVRC", "Data", "DET", "train", "ILSVRC2013_train")
#IMAGENET_DIR =  os.path.join(IMAGE_DIR, "ILSVRC2013_train")

df = pd.read_csv(os.path.join(DATA_DIR, "synsets_dog_cat.csv"), names=["synset"], encoding="utf-8", header=1)
synsets = df.values

valid = []

for row in synsets:
    synset = row[0]    
    synset_path = os.path.join(IMAGENET_DIR, synset)
    if os.path.isdir(synset_path):
        valid.append(synset)



data = pd.DataFrame(valid,columns=['synset'])
data.to_csv(os.path.join(DATA_DIR, "synsets_dog_cat_valid.csv"))

print("Salvo em ", os.path.join(DATA_DIR, "synsets_dog_cat_valid.csv"))