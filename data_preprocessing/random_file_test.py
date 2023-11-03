import os
import random

# Image Directory
directory = "./data/test/transportation"

# List all jpg files
image_files = [
    f
    for f in os.listdir(directory)
    if f.endswith(".jpg") or f.endswith(".png")
]
# sort based on integer before '.jpg'
image_files.sort(key=lambda x: int(x.split(".")[0]))

# Shuffle the list
random.shuffle(image_files)

# Rename each file with a new index
for idx, filename in enumerate(image_files):
    old_path = os.path.join(directory, filename)
    new_name = f"temp_{idx}.jpg"
    new_path = os.path.join(directory, new_name)
    os.rename(old_path, new_path)

# List all temporary files and shuffle
temp_files = [f for f in os.listdir(directory) if f.startswith("temp_")]
random.shuffle(temp_files)

# Rename each temporary file to a new index
for idx, temp_filename in enumerate(temp_files):
    old_path = os.path.join(directory, temp_filename)
    new_name = f"{idx}.jpg"
    new_path = os.path.join(directory, new_name)
    os.rename(old_path, new_path)
print("Shuffling and renaming completed!")
