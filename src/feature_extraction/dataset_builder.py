import os
import csv
from .image_processing import load_image, segment_pistachio
from src.feature_extraction.feature_extraction import extract_all_features

def create_dataset_csv(base_path, output_csv="pistachio_features.csv"):
    feature_names = ['AREA', 'PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS', 'ECCENTRICITY', 
                    'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA', 'EXTENT', 'ASPECT_RATIO', 
                    'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_1', 'SHAPEFACTOR_2', 
                    'SHAPEFACTOR_3', 'SHAPEFACTOR_4', 'CLASS']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(feature_names)
        total_processed = 0

        for class_name in ["Kirmizi_Pistachio", "Siirt_Pistachio"]:
            class_path = os.path.join(base_path, class_name)
            for filename in os.listdir(class_path):
                if filename.endswith(('.jpg', '.png')):
                    try:
                        img_rgb, img_gray = load_image(os.path.join(class_path, filename))
                        binary = segment_pistachio(img_gray)
                        features = extract_all_features(img_rgb, binary)
                        features.append(class_name)
                        writer.writerow(features)
                        total_processed += 1
                        
                        if total_processed % 100 == 0:
                            print(f"Procesadas: {total_processed}")
                    except Exception as e:
                        print(f"Error con {filename}: {e}")

    print(f"¡Completado! Total procesadas: {total_processed}")
    return output_csv
