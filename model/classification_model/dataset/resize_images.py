from PIL import Image
import os

input_folder = "images"
output_folder = "resized_images"
target_size = (128, 128)
quality = 85


def resize_images(input_dir, output_dir, size, quality=85):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                input_path = os.path.join(root, filename)

                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with Image.open(input_path) as img:
                    resized_img = img.resize(size)
                    if filename.lower().endswith(".png"):
                        resized_img.save(output_path)
                    else:
                        resized_img.save(output_path, quality=quality)

                print(f"{filename} — done")


resize_images(input_folder, output_folder, target_size, quality)
