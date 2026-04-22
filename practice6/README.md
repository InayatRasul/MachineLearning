# ML FastAPI Docker Project

## Run locally
uvicorn main:app --reload

## Build Docker
docker build -t ml-fastapi .

## Run Docker
docker run -p 8000:8000 ml-fastapi

## API Docs
http://localhost:8000/docs