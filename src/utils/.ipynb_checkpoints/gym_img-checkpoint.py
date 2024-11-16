from PIL import Image
import os

def image():
    print("============================================================")
    print(os.getcwd())
    print("============================================================")
    print(os.path.join(os.getcwd(), "src/image/노이탈(헬스)2.webp"))
    image_path_1 = os.path.join(os.getcwd(), "src/image/노이탈(헬스)2.webp")
    image_path_2 = os.path.join(os.getcwd(), "src/image/이탈자(헬스).webp")
    image_1 = Image.open(image_path_1)
    image_2 = Image.open(image_path_2)
    return (image_1, image_2)
