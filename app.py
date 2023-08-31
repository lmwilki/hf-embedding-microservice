from sentence_transformers import SentenceTransformer
import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from typing import List

available_models = []

for dirpath, dirnames, filenames in os.walk("models"):
    # Check if the current directory contains a file named "config.json"
    if "pytorch_model.bin" in filenames:
        # If it does, print the path to the directory
        available_models.append(dirpath)

models = {}

for model_path in available_models:
    model_name = model_path.split("/")[-1]
    models[model_name] = SentenceTransformer(model_path)
    print(f"Loaded model {model_name}")


app = FastAPI(
    title="Sentence Embedding API",
    description="A simple API that uses sentence-transformers to embed sentences"
)

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/models")
def get_models():
    return {"models": list(models.keys())}

@app.get("/models/{model_name}")
def get_model(model_name: str):
    if model_name not in models:
        return {"message": f"Model {model_name} not found"}
    return True

@app.post("/models/{model_name}/embed")
def embed_sentences(model_name: str, sentences: List[str]):

    if model_name not in models:
        return {"message": f"Model {model_name} not found"}
    
    model = models[model_name]
    embeddings = model.encode(sentences)
    return {"embeddings": embeddings.tolist()}

@app.post("/models/{model_name}/similarity")
def similarity(model_name: str, sentences: List[str]):
    if model_name not in models:
        return {"message": f"Model {model_name} not found"}
    
    model = models[model_name]
    embeddings = model.encode(sentences)
    similarity_matrix = [[round(float(x), 4) for x in embeddings[i].dot(embeddings.T)] for i in range(len(sentences))]
    return {"similarity_matrix": similarity_matrix}







