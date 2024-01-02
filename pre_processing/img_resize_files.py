from lib_pre_proces import *

val = 0
target_size = 64

# Directory with images
image_dir = image_dir + str(val)
# Target directory to save processed images
output_dir = image_dir

# List all image files in the directory
image_files = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
print("Total image files:", len(image_files))

# Create a list to store all processed NumPy arrays
all_image_arrays = []

# Load and process images in the batch
for image_file in image_files:
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

    # Save processed image with "_pp" suffix and the same name
    original_shape_img = processed_image_array.reshape(target_size, target_size)
    pil_img = Image.fromarray(original_shape_img)
    output_filename = os.path.join(output_dir, os.path.basename(image_file).replace('.jpg', '_pp.jpg'))
    pil_img.save(output_filename)

csv_file_path = 'images.csv'

# Open the file and write number y on each line
with open(csv_file_path, 'a', newline='') as csvfile:
    for _ in range(target_width):
        csvfile.write(f"{val}\n")

print("Finished processing all batches and saved images to images.csv.")
