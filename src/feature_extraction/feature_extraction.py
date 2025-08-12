import cv2
import numpy as np

def extract_area(binary_img):
    return np.sum(binary_img == 255)

def extract_perimeter(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.arcLength(largest_contour, True)

def extract_axes(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(largest_contour)
    _, (width, height), _ = ellipse
    return max(width, height), min(width, height)

def extract_eccentricity(major_axis, minor_axis):
    if major_axis == 0:
        return 0
    return np.sqrt(1 - (minor_axis/major_axis)**2)

def extract_solidity_convex_area(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    convex_area = cv2.contourArea(hull)
    object_area = cv2.contourArea(largest_contour)
    solidity = object_area / convex_area if convex_area > 0 else 0
    return solidity, convex_area

def extract_all_features(img_rgb, binary_img):
    area = extract_area(binary_img)
    perimeter = extract_perimeter(binary_img)
    major_axis, minor_axis = extract_axes(binary_img)
    eccentricity = extract_eccentricity(major_axis, minor_axis)
    solidity, convex_area = extract_solidity_convex_area(binary_img)
    
    eqdiasq = np.sqrt(4 * area / np.pi)
    extent = area / (img_rgb.shape[0] * img_rgb.shape[1])
    aspect_ratio = major_axis / minor_axis if minor_axis != 0 else 0
    roundness = (4 * np.pi * area) / (perimeter**2) if perimeter != 0 else 0
    compactness = (perimeter**2) / area if area != 0 else 0
    
    shapefactor_1 = roundness
    shapefactor_2 = (np.pi * (major_axis/2)**2) / area if area != 0 else 0  
    shapefactor_3 = major_axis / area if area != 0 else 0
    shapefactor_4 = area / (major_axis**2) if major_axis != 0 else 0
    
    return [area, perimeter, major_axis, minor_axis, eccentricity, eqdiasq, 
            solidity, convex_area, extent, aspect_ratio, roundness, compactness,
            shapefactor_1, shapefactor_2, shapefactor_3, shapefactor_4]
