import time

import cv2
import numpy as np
from PIL import Image

from unet import Unet_ONNX, Unet

if __name__ == "__main__":
    mode = "predict"
    count           = False
    name_classes    = ["background","NF1"]
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    test_interval = 100
    fps_image_path  = "img/1001.jpg"
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    simplify        = True

    if mode != "predict_onnx":
        unet = Unet()
    else:
        yolo = Unet_ONNX()

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

        
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = unet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    
