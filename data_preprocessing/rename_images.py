# Becareful when using this. Only run 1 time for split dataset to train, validation and test
# UNCOMMENT THE CODE BELOW:
# import os

# # Change the collection to your data folder
# collection = "./data/test"
# for i, filename in enumerate(os.listdir(collection)):
#     old_file_path = os.path.join(collection, filename)
#     new_file_name = str(i) + ".jpg"
#     new_file_path = os.path.join(collection, new_file_name)
#     os.rename(old_file_path, new_file_path)
