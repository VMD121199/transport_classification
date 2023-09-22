import os, random, shutil

dir_path = "./data//train/Bicycle"
destination_path = "./data/test/Bicycle"

filenames = random.sample(os.listdir(dir_path), 400)
for fname in filenames:
    srcpath = os.path.join(dir_path, fname)
    shutil.move(srcpath, destination_path)