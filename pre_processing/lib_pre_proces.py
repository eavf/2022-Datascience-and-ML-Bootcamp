import os
import csv
from PIL import Image, ImageOps
import numpy as np

target_width = 64
target_height = 64
target_size = 64

# Directory with images
image_dir = '../SEDS/250000_Final/'


def zapis_subor(x_path, x_data, y_path, dlzka, y):
    with open(x_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for image_array in x_data:
            # Flatten the array and write each element as a separate column
            writer.writerow(image_array.flatten())

    # Open the file and write 0 on each line
    with open(y_path, 'a', newline='') as csvfile:
        for _ in range(dlzka):
            csvfile.write(f"{y}\n")


def process(image_array):
    # Process the image array (e.g., set values less than 80 to 0)
    return np.where(image_array < 125, 0, image_array)


def najdi_max_dim():
    # Prechádzanie podadresárov od 0 po 9
    img_dim = [0, 0]
    for subdir in range(10):
        subdir_path = os.path.join(image_dir, str(subdir))
        print("som v podadresári : ", subdir_path)

        # List all image files in the directory
        image_files = [os.path.join(subdir_path, filename) for filename in os.listdir(subdir_path) if
                       filename.endswith('.jpg')]
        print("Total image files:", len(image_files))

        # Load and process images in the batch
        for image_file in image_files:
            img_adr = img_dim
            img = Image.open(image_file)
            # Zistenie veľkosti obrázka
            width, height = img.size
            if img_adr[0] < width:
                img_adr[0] = width
                namew = image_file
            if img_adr[1] < height:
                img_adr[1] = height
                nameh = image_file

    print(f"Najväčší rozmer obrázka celkom : width : {img_dim[0]} v súbore {namew} a height: {img_dim[1]} v súbore {nameh}")
    return img_dim