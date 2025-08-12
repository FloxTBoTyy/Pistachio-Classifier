from src.feature_extraction.dataset_builder import create_dataset_csv
from src.modeling.model_training import train_with_complete_evaluation

if __name__ == "__main__":
    base_path = "data/Pistachio_Image_Dataset"
    csv_file = create_dataset_csv(base_path)   # Crea o actualiza dataset CSV
    train_with_complete_evaluation(csv_file)  # Entrena y evalúa el modelo
