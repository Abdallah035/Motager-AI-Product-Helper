# Motager AI Product Generator  

**Motager AI Product Generator** is an AI-powered system designed to automate product content generation for the **Motager** e-commerce platform. This solution leverages machine learning to extract colors from product images, generate relevant product names, and create detailed descriptions, streamlining the product listing process.

## 🚀 Features  

- 🎨 **AI-Powered Color Extraction** – Detects and extracts colors from product images.  
- 🏷 **Product Name Generation** – Generates contextually relevant product names.  
- 📝 **Detailed Product Description Generation** – Creates engaging and informative descriptions for e-commerce.  
- ⚡ **FastAPI Backend** – Provides structured and efficient API endpoints for seamless integration.  

## ⚙️ API Endpoints  

The system is built using **FastAPI** and offers three key endpoints:  

### 1️⃣ **Extract Colors from Image**  
- **Endpoint:** `POST /extract-color`  
- **Description:** Extracts all colors from a product image.  
- **Request:**  
  ```json
  {
    "image_url": "https://example.com/product-image.jpg"
  }
