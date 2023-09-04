from sentence_transformers import SentenceTransformer
import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from typing import List
from uuid import uuid4
import pydantic
from datetime import datetime


# === Data Models ===

class ModelInfo(pydantic.BaseModel):
    model_name: str
    max_seq_length: int
    vector_size: int


class EmbeddingResponse(pydantic.BaseModel):
    embedding_id: str
    model_info: ModelInfo
    embeddings: List[List[float]]

class EmbeddingRequest(pydantic.BaseModel):
    sentences: List[str]

class ModelListResponse(pydantic.BaseModel):
    models: List[ModelInfo]

class StatusResponse(pydantic.BaseModel):
    status: str
    uptime_seconds: float
    version: str


# === API Setup ===

start_time = datetime.now()
IS_READY = False

available_models = []

for dirpath, dirnames, filenames in os.walk("models"):
    # Check if the current directory contains a file named "config.json"
    if "pytorch_model.bin" in filenames:
        # If it does, print the path to the directory
        available_models.append(dirpath)

# Load all models
models = {}

for model_path in available_models:
    model_name = model_path.split("/")[-1]
    models[model_name] = SentenceTransformer(model_path)
    print(f"Loaded model {model_name}")

# Create model info
model_info = {}

for model_name, model in models.items():

    model_info_entry = {
        "model_name": model_name,
        "max_seq_length": model.get_max_seq_length(),
        "vector_size": model.get_sentence_embedding_dimension()
    }

    model_info[model_name] = model_info_entry

IS_READY = True

# Create API

app = FastAPI(
    title="Sentence Embedding API",
    description="A simple API that uses sentence-transformers to embed sentences",
    version="0.1.0",
    openapi_tags=[
        {
            "name": "models",
            "description": "Get information about the available models"
        },
        {
            "name": "health",
            "description": "Health check"
        }
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# === API Routes ===

# Basic Setup

@app.get("/", include_in_schema=False, response_class=RedirectResponse)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=StatusResponse, tags=["health"])
def health():
    """Returns the health of the API

    Returns:
        StatusResponse: The health of the API
    """

    uptime = datetime.now() - start_time
    uptime_seconds = uptime.total_seconds()

    output = {
        "status": None,
        "uptime_seconds": uptime_seconds,
        "version": app.version
    }

    if IS_READY:
        output["status"] = "ready"
    else:
        output["status"] = "loading"
    
    return output

# Models and Embeddings

@app.get("/models", response_model=ModelListResponse, tags=["models"])
def get_models():
    """Returns a list of available models

    Returns:
        ModelListResponse: A list of available models
    """
    return {"models": list(model_info.values())}

@app.get("/models/{model_name}", response_model=ModelInfo, tags=["models"])
def get_model(model_name: str):
    """Returns information about a given model

    Args:
        model_name (str): The name of the model

    Returns:
        ModelInfo: Information about the model
    """

    if model_name not in models:
        return {"message": f"Model {model_name} not found"}
    return model_info[model_name]

@app.post("/models/{model_name}/embed", response_model=EmbeddingResponse, tags=["models"])
def embed_sentences(model_name: str, sentences: List[str]):
    """Embeds a list of sentences using a given model

    Args:
        model_name (str): The name of the model
        sentences (List[str]): A list of sentences

    Returns:
        EmbeddingResponse: The embeddings of the sentences
    """

    if model_name not in models:
        return {"message": f"Model {model_name} not found"}
    
    model = models[model_name]
    embeddings = model.encode(sentences)

    output = {
        "embedding_id": str(uuid4()),
        "model_info": model_info[model_name],
        "embeddings": embeddings
    }

    return output







