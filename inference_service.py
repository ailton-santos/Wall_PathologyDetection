" Wall Pathology Detection - By Ailton Dos Santos"
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import numpy as np


from core_model import CivilPathologyModel


class PathologyPredictor:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- 🇩🇪 Initializing Logic on {self.device} ---")

        # Initialize the Model
        self.model = CivilPathologyModel(num_classes=2)

        # Move to CPU/GPU
        self.model.to(self.device)
        self.model.eval()

        # Define the Image Transformations (Resize -> Normalize)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_embedding(self, image_bytes):
        """Helper to get the math vector from image bytes"""
        # Open image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # Transform to tensor
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Get embedding from model
        with torch.no_grad():
            embedding = self.model(tensor)
        return embedding

    def compare_two_walls(self, reference_bytes, sample_bytes):
        """
        Compare a 'Known Good' wall (Reference) vs a 'Suspect' wall (Sample).
        This distinguishes 'Dirty' (High Similarity) from 'Broken' (Low Similarity).
        """
        # 1. Get Embeddings for both images
        vec_ref = self._get_embedding(reference_bytes)
        vec_sample = self._get_embedding(sample_bytes)

        # 2. Calculate Cosine Similarity
        # Result is between -1.0 (opposite) and 1.0 (identical)
        similarity = torch.mm(vec_ref, vec_sample.t()).item()

        # 3. Smart Thresholding
        # > 0.80 : Likely just dirty or different lighting (HEALTHY)
        # < 0.80 : Structural difference (PATHOLOGY)
        threshold = 0.80

        is_pathology = similarity < threshold

        # Interpretation Logic
        status_msg = "HEALTHY_STRUCTURE"
        if is_pathology:
            status_msg = "POTENTIAL_PATHOLOGY_DETECTED"

        return {
            "status": "success",
            "analysis_type": "COMPARATIVE_INSPECTION",
            "similarity_score": round(similarity, 4),
            "prediction": status_msg,
            "interpretation": "Similar Texture (Dirty/Shadow)" if similarity >= threshold else "Structural Mismatch (Crack/Detachment)"
        }


    def predict_image(self, image_bytes):
        pass