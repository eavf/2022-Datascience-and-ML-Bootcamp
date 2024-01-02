from PIL import Image, ImageOps
import numpy as np
import os
from lib_pre_proces import *


# Batch size
batch_size = 100

# Files with datasets
csv_train_path = '../datas/SEDS/old/x_train.csv'
csv_test_path = '../datas/SEDS/old/x_test.csv'
csv_val_path = '../datas/SEDS/old/x_val.csv'

# Prechádzanie podadresárov od 0 po 9
for subdir in range(10):
    subdir_path = os.path.join(image_dir, str(subdir))
    subdir_path1 = os.path.join('../SEDS/250000_Final/', str(subdir))
    print("som v podadresári : ", subdir_path)

    # Create a list to store all processed NumPy arrays
    all_image_arrays = []

    # List all image files in the directory
    image_files = [os.path.join(subdir_path, filename) for filename in os.listdir(subdir_path) if
                   filename.endswith('.jpg')]
    print("Total image files:", len(image_files))

    # Number of batches
    num_batches = len(image_files) // batch_size
    print("Number of batches : ", num_batches)

    for batch_num in range(num_batches + 1):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(image_files))  # Avoid going beyond list
        batch_image_files = image_files[start_idx:end_idx]

        # Load and process images in the batch
        for image_file in batch_image_files:
            image = Image.open(image_file)
            width, height = image.size
            if image.size[0] > image.size[1]:
                # Prvý stupen redimenzovania
                wpercent = (target_size / float(image.size[0]))
                hsize = int((float(image.size[1]) * float(wpercent)))
                resized_image = image.resize((target_size, hsize), Image.Resampling.LANCZOS)
            else:
                # Prvý stupen redimenzovania
                hpercent = (target_size / float(image.size[1]))
                wsize = int((float(image.size[0]) * float(hpercent)))
                resized_image = image.resize((wsize, target_size), Image.Resampling.LANCZOS)

            # Druhý stupeň redimenzobvania bude robený z np array
            bw_processed_image = resized_image.convert('L')

            # Konvertovanie PIL Image na NumPy array
            image_np = np.array(bw_processed_image)

            # Calculate the amount of padding needed on the top, bottom, left, and right sides
            top_pad = max(0, (target_size - image_np.shape[0]) // 2)
            bottom_pad = max(0, target_size - image_np.shape[0] - top_pad)
            left_pad = max(0, (target_size - image_np.shape[1]) // 2)
            right_pad = max(0, target_size - image_np.shape[1] - left_pad)

            # Use numpy.pad to add padding and fill with 255
            image_np = np.pad(
                image_np,
                pad_width=((top_pad, bottom_pad), (left_pad, right_pad)),
                mode='constant',
                constant_values=255  # Fill with 255 (white)
            )

            image_array = np.invert(image_np)
            image_array = image_array.ravel()

            image_array = image_array.reshape(1, -1)
            processed_image_array = process(image_array)

            # Append the processed NumPy array to the list
            all_image_arrays.append(processed_image_array)

            """
            # Save processed image with "_pp" suffix and the same name
            original_shape_img = processed_image_array.reshape(target_size, target_size)
            pil_img = Image.fromarray(original_shape_img)
            output_filename = os.path.join(subdir_path, os.path.basename(image_file).replace('.jpg', '_pp.jpg'))
            pil_img.save(output_filename)
            """

    # Rozdelenie na train, test and val:
    # Calculate lengths for each segment
    total_images = len(all_image_arrays)
    train_size = int(0.6 * total_images)
    test_size = int(0.2 * total_images)
    # The remaining 20% will go to validation

    # Split the array
    x_train = all_image_arrays[:train_size]
    x_test = all_image_arrays[train_size:train_size + test_size]
    x_val = all_image_arrays[train_size + test_size:]

    # Save processed images to a CSV file (one image per row)
    print(f"Spracovávam adresár s číslom {subdir}")

    # Train dataset
    zapis_subor(csv_train_path, x_train, '../datas/SEDS/old/y_train.csv', train_size, subdir)
    print(f"Added {train_size} records to the x_train file")

    # Test dataset
    zapis_subor(csv_test_path, x_test, '../datas/SEDS/old/y_test.csv', test_size, subdir)
    print(f"Added {test_size} records to the x_test file")

    # Val dataset
    zapis_subor(csv_val_path, x_val, '../datas/SEDS/old/y_val.csv', total_images - train_size - test_size, subdir)
    print(f"Added {total_images - train_size - test_size} records to the x_val file")

print("Finished processing all batches and saved images to images.csv.")
