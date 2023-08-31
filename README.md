# hf-embedding-microservice

## Description
A sentence embedding microservice that downloads models during build time and serves them via a REST API. Downloading saves time when horizontally scaling the service so it can be deployed on multiple machines without re-downloading the models.



## Usage
### Build
```bash
docker build -t hf-embedding-microservice:latest . 
```

You can swap the model by changing the `MODEL_NAME` environment variable in the Dockerfile. The default model is `sentence-transformers/all-mpnet-base-v2`. You can find a list of models [here](https://huggingface.co/sentence-transformers).

```bash
docker build -t hf-embedding-microservice:latest . -e MODEL_NAME="sentence-transformers/paraphrase-MiniLM-L6-v2"
```

### Run
```bash
docker run -p 5000:5000 hf-embedding-microservice
```

### Docker Compose
```bash
docker-compose up
```

### Access the API
You can load the docs at http://localhost:5000/docs or http://localhost:5000 (it redirects to /docs).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT License](https://choosealicense.com/licenses/mit/)