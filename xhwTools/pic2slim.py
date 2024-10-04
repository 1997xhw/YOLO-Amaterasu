import os
from PIL import Image
import shutil


def resize_and_crop_image(input_path, output_path, target_width=1280, target_height=720):
    """
    Resize an image and crop it to the target width and height.
    Args:
    - input_path: Path to the input image.
    - output_path: Path where the resized and cropped image will be saved.
    - target_width: Target width of the image.
    - target_height: Target height of the image.
    """
    with Image.open(input_path) as img:
        # Calculate the target aspect ratio
        target_ratio = target_width / target_height
        img_ratio = img.width / img.height

        # Determine if the image needs to be cropped width-wise or height-wise
        if img_ratio > target_ratio:
            # Crop width to match the target ratio
            new_width = int(target_ratio * img.height)
            left = (img.width - new_width) / 2
            right = left + new_width
            img = img.crop((left, 0, right, img.height))
        elif img_ratio < target_ratio:
            # Crop height to match the target ratio
            new_height = int(img.width / target_ratio)
            top = (img.height - new_height) / 2
            bottom = top + new_height
            img = img.crop((0, top, img.width, bottom))

        # Resize the image
        img = img.resize((target_width, target_height), Image.LANCZOS)
        img.save(output_path)


def resize_images_wt_in_folder(folder_path, folder_output_path, target_width=1280, target_height=720):
    """
    Resize and crop all images in the specified folder to the given dimensions.
    Args:
    - folder_path: Folder containing the images to process.
    - target_width: Target width of the images.
    - target_height: Target height of the images.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(folder_output_path, f"{filename}")
            resize_and_crop_image(input_path, output_path, target_width, target_height)
            # print(f"Processed {filename}")


# 改变原图大小
def resize_images_in_folder(folder, to_folder_path, max_size=100000, exts=['.jpg', '.jpeg', '.png']):
    """
    Resize images in the folder to ensure each is under the specified max size in bytes.
    Args:
    - folder: Folder containing images.
    - max_size: Maximum file size in bytes.
    - exts: List of acceptable image extensions.
    """
    for img_filename in os.listdir(folder):
        if any(img_filename.lower().endswith(ext) for ext in exts):
            img_path = os.path.join(folder, img_filename)
            with Image.open(img_path) as img:
                # Calculate the target size
                scale_factor = (max_size / os.path.getsize(img_path)) ** 0.5
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)

                # Resize the image and save
                img = img.resize((new_width, new_height), Image.LANCZOS)
                new_img_path = os.path.join(to_folder_path, f"{img_filename}")
                img.save(new_img_path, quality=85)  # Adjust quality as needed
                print(f"Resized {img_filename} and saved as {new_img_path}")


# 重命名文件
def rename_images(folder_path, start_index=1569, file_exts=['.jpg', '.jpeg', '.png', '.txt']):
    """
    Rename images in the folder starting from a given index with padded numbers.
    Args:
    - folder_path: Folder containing images.
    - start_index: The starting index for numbering.
    - file_exts: List of acceptable image extensions.
    """
    files = [file for file in os.listdir(folder_path) if any(file.lower().endswith(ext) for ext in file_exts)]
    files.sort()  # Sort files to maintain order, optional, depends on specific needs

    # Pad the index numbers to maintain a consistent file name length
    pad_length = len(str(len(files) + start_index - 1))

    for index, filename in enumerate(files, start=start_index):
        new_name = f"{str(index).zfill(pad_length)}.jpg"  # Assuming the images are in JPEG format
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        shutil.move(old_path, new_path)
        print(f"Renamed {filename} to {new_name}")


if __name__ == '__main__':
    # Use the function
    # folder_path = 'D:\\YoloData'
    folder_path = r"D:\yolopm-all\TP-paving\finished\images"
    # to_folder_path = r"D:\yolopm-all\TP-paving\images-slim"
    # resize_images_wt_in_folder(folder_path, to_folder_path)
    # resize_images_in_folder(folder_path, to_folder_path)
    rename_images(folder_path)
