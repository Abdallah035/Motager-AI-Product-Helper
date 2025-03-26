from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from Generate_caption import load_model_from_path, tokenizer_load
from Color_extraction import extract_colors
from Generate_productName_description import generate_product_name, generate_description

# Initialize FastAPI
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not set. Please configure your .env file or system environment.")

# Ensure ONNX model path is set
os.environ["XDG_CACHE_HOME"] = "models/u2net.onnx"

# Global variables for models
vgg16_model = None
fifth_version_model = None
tokenizer = None


async def load_models():
    global vgg16_model, fifth_version_model, tokenizer
    print("Loading models concurrently...")
    vgg16_task = asyncio.create_task(asyncio.to_thread(load_model_from_path, 'models/vgg16_feature_extractor.keras'))
    fifth_version_task = asyncio.create_task(
        asyncio.to_thread(load_model_from_path, 'models/fifth_version_model.keras'))
    tokenizer_task = asyncio.create_task(asyncio.to_thread(tokenizer_load, 'models/tokenizer.pkl'))
    vgg16_model, fifth_version_model, tokenizer = await asyncio.gather(
        vgg16_task, fifth_version_task, tokenizer_task
    )
    print("Models loaded successfully!")


# Run model loading at startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_models())


# Pydantic Models
class ImagePathsRequest(BaseModel):
    image_paths: List[str]


class GenerateProductRequest(ImagePathsRequest):
    Brand_name: str


class GenerateDescriptionRequest(BaseModel):
    product_name: str
    colors: Optional[List[str]] = None


class AIproducthelper(ImagePathsRequest):
    Brand_name: str
    colors: Optional[List[str]] = None


# Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Internal Server Error", "code": 500, "error": repr(exc)},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": exc.detail, "code": exc.status_code, "error": repr(exc)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"success": False, "message": "Validation Error", "code": 422, "error": exc.errors()},
    )


@app.get("/")
async def read_root():
    return {"message": "Hello from our API , All models are loaded successfully!"}

@app.get("/status/")
async def check_status():
    global vgg16_model, fifth_version_model, tokenizer
    if vgg16_model and fifth_version_model and tokenizer:
        return {"success": True, "message": "Models are ready!"}
    else:
        return {"success": False, "message": "Models are still loading..."}

@app.post("/extract-colors/")
async def extract_colors_endpoint(request: ImagePathsRequest):
    if not request.image_paths:
        raise HTTPException(status_code=500, detail="Internal Server Error: Image list cannot be empty.")

    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            colors = await loop.run_in_executor(executor, extract_colors, request.image_paths)

        return {"success": True, "colors": colors}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {repr(exc)}")


@app.post("/generate-product-name/")
async def generate_product_name_endpoint(request: GenerateProductRequest):
    if not request.image_paths:
        raise HTTPException(status_code=500, detail="Internal Server Error: Image list cannot be empty.")

    try:
        product_name = await asyncio.to_thread(
            generate_product_name, request.image_paths, request.Brand_name, vgg16_model, fifth_version_model, tokenizer,
            API_KEY
        )
        return {"success": True, "product_name": product_name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {repr(exc)}")


@app.post("/generate-description/")
async def generate_description_endpoint(request: GenerateDescriptionRequest):
    try:
        description = await asyncio.to_thread(
            generate_description, API_KEY, request.product_name, vgg16_model, fifth_version_model, tokenizer,
            request.colors
        )
        return {"success": True, "description": description}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {repr(exc)}")


@app.post("/AI-product_help/")
async def ai_product_help_endpoint(request: AIproducthelper):
    if not request.image_paths:
        raise HTTPException(status_code=500, detail="Internal Server Error: Image list cannot be empty.")

    try:
        product_name = await asyncio.to_thread(
            generate_product_name, request.image_paths, request.Brand_name, vgg16_model, fifth_version_model, tokenizer,
            API_KEY
        )
        description = await asyncio.to_thread(
            generate_description, API_KEY, product_name, vgg16_model, fifth_version_model, tokenizer, request.colors
        )
        return {
            "success": True,
            "product_name": product_name,
            "description": description
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {repr(exc)}")
