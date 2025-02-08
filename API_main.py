from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from fastapi.exceptions import RequestValidationError
from dotenv import load_dotenv
import os
from PIL import Image
from Color_extraction import extract_colors
from Generate_productName_description import generate_product_name, generate_description

app = FastAPI()

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not set. Please configure your .env file or system environment.")


# Models
class Extract_colors(BaseModel):
    image_paths: List[str]


class Generate_product_name_item(BaseModel):
    image_paths: List[str]


class Generate_description_item(BaseModel):
    image_paths: List[str]
    product_name: str
    colors: Optional[List[str]] = None


# Custom exception handler for general exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected error occurred.",
            "code": 500,
            "error": repr(exc)
        },
    )


# Custom exception handler for HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "code": exc.status_code,
            "error": repr(exc)
        },
    )

# Custom exception handler for RequestValidationError
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation error occurred.",
            "code": 422,
            "error": exc.errors()
        },
    )

# test for api
@app.get("/")
def read_root():
    return {"message": "Hello from our API"}

#verify the image paths amd image itself
def verify_image_paths(image_paths: List[str]) -> None:
    for path in image_paths:
        if not os.path.exists(path):
            raise HTTPException(
                status_code=400,
                detail=f"There is no file found at path: {path}"
            )
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"The file at path {path} is not a valid image."
            )


@app.post('/extract-colors/')
async def extract_colors_endpoint(request: Extract_colors):
    try:
        if not request.image_paths:
            raise HTTPException(
                status_code=400,
                detail="The image list cannot be empty."
            )
        verify_image_paths(request.image_paths)
        colors = extract_colors(request.image_paths)
        return {"success": True, "colors": colors}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise e  # Catch all will be handled by global_exception_handler


@app.post('/generate-product-name/')
async def generate_product_name_endpoint(request: Generate_product_name_item):
    try:
        if not request.image_paths:
            raise HTTPException(
                status_code=400,
                detail="The image list cannot be empty."
            )
        verify_image_paths(request.image_paths)
        # API_KEY is used internally by the server
        product_name = generate_product_name(request.image_paths, API_KEY)
        return {"success": True, "product_name": product_name}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))  # Example of bad input
    except Exception as e:
        raise e

@app.post('/generate-description/')
async def generate_description_endpoint(request: Generate_description_item):
    try:
        if not request.image_paths:
            raise HTTPException(
                status_code=400,
                detail="The image list cannot be empty."
            )
        verify_image_paths(request.image_paths)
        description = generate_description(
            request.image_paths, API_KEY, request.product_name, request.colors)
        return {"success": True, "description": description}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))  # Specific bad request error
    except Exception as e:
        raise e
