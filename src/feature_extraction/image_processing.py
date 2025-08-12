import cv2

def load_image(image_path):
    """Carga una imagen y la prepara para extracción de características"""
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return img_rgb, img_gray

def segment_pistachio(img_gray):
    """Segmenta el pistacho del fondo negro"""
    import numpy as np
    _, binary = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary
