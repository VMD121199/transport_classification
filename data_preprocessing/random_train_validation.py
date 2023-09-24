# Becareful when using this. Only run 1 time for split dataset to train, validation and test
# UNCOMMENT THE CODE BELOW:
# import os, random, shutil

# dir_path = "./data//validation/Bicycle"
# destination_path = "./data/train/Bicycle"

# filenames = random.sample(os.listdir(dir_path), 200)
# for fname in filenames:
#     srcpath = os.path.join(dir_path, fname)
#     shutil.move(srcpath, destination_path)
