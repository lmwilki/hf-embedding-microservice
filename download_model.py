from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = os.getenv("MODEL_NAME", "all-mpnet-base-v2")

models_path = f"models/"
model = SentenceTransformer(MODEL_NAME, cache_folder=models_path)