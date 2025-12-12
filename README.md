# Wall Pathology Detection System 
By Ailton Dos Santos

## Overview
This project is a **Deep Learning-based Structural Health Monitoring (SHM)** system designed to detect civil engineering pathologies (such as cracks and material detachment) in concrete walls.

Unlike standard classifiers, this system utilizes a **Metric Learning approach (ArcFace)** combined with a **ResNet50** backbone. This allows the model to learn highly discriminative features, enabling it to distinguish between **actual structural damage** versus cosmetic issues like **dirt, shadows, or humidity stains** via cosine similarity comparison.

## Key Features
* **ArcFace Loss Layer:** Implements Large Margin Arc Distance for superior feature embedding separation.
* **Comparative Inspection (One-Shot Logic):** Compares a "Reference" (Healthy) image against a "Suspect" image to calculate a similarity score.
* **Robustness:** Successfully ignores surface dirt to focus on structural integrity.
* **REST API:** Built with **FastAPI** for high-performance, real-time inference.

## Tech Stack
* **Core:** Python 3.9+, PyTorch, Torchvision
* **Architecture:** ResNet50 (Pretrained) + Custom ArcFace Head
* **API Framework:** FastAPI + Uvicorn
* **Image Processing:** PIL (Pillow), NumPy

## How it Works
The system exposes a comparison endpoint where two images are uploaded:
1.  **Reference Image:** A known healthy section of the wall.
2.  **Suspect Image:** The area being inspected.

The model extracts 2048-dimensional embedding vectors for both images. If the **Cosine Similarity** drops below the threshold (0.80), the system flags a **Potential Pathology** (Structural Mismatch). If the score is high, it classifies the variance as texture/dirt.

## Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ailton-santos/Wall_PathologyDetection.git](https://github.com/ailton-santos/Wall_PathologyDetection.git)
    cd Wall_PathologyDetection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the API Server:**
    ```bash
    python main.py
    ```

4.  **Test the Endpoint:**
    Open your browser at `http://localhost:8000/docs` to use the interactive Swagger UI.
    * Use the **POST /compare/walls** endpoint.
    * Upload a reference image and a damaged image to see the similarity score.

## Project Structure
* `core_model.py`: Defines the Neural Network architecture (ResNet50 + ArcFaceLayer).
* `inference_service.py`: Handles image preprocessing, embedding extraction, and similarity logic.
* `main.py`: The FastAPI application entry point.
