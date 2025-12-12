" Wall Pathology Detection - By Ailton Dos Santos"
from fastapi import FastAPI, File, UploadFile, HTTPException
from inference_service import PathologyPredictor
import uvicorn

app = FastAPI(title="Wall Pathology Detection", version="1.1.0")
predictor = PathologyPredictor()


@app.get("/")
def home():
    return {"message": "System Online. Use /compare/walls to test."}


# NEW ENDPOINT: Upload Reference AND Sample
@app.post("/compare/walls")
async def compare_structures(
        reference_image: UploadFile = File(..., description="A photo of a CLEAN, HEALTHY section of the wall."),
        suspect_image: UploadFile = File(..., description="The photo of the DIRTY or DAMAGED section.")
):
    """
    Compares a suspect wall against a known healthy reference.
    This distinguishes 'Dirty' (High Sim) from 'Broken' (Low Sim).
    """
    # Validate
    if reference_image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "Reference must be JPEG/PNG")

    # Read Files
    ref_bytes = await reference_image.read()
    sample_bytes = await suspect_image.read()

    # Run Logic
    result = predictor.compare_two_walls(ref_bytes, sample_bytes)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)