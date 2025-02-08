# Motager AI Product Generator  

**Motager AI Product Generator** is an AI-powered system designed to automate product content generation for the **Motager** e-commerce platform. This solution leverages machine learning to extract colors from product images, generate relevant product names, and create detailed descriptions, streamlining the product listing process.

---

## 🚀 Features  

- 🎨 **AI-Powered Color Extraction** – Detects and extracts colors from product images.  
- 🏷 **Product Name Generation** – Generates contextually relevant product names.  
- 📝 **Detailed Product Description Generation** – Creates engaging and informative descriptions for e-commerce.  
- ⚡ **FastAPI Backend** – Provides structured and efficient API endpoints for seamless integration.  

---

## ⚙️ API Endpoints  

The system is built using **FastAPI** and offers three key endpoints:  

### 1️⃣ **Extract Colors from Image**  
- **Endpoint:** `POST /extract-color`  
- **Description:** Extracts all colors from a product image.  
- **Request:**  
  ```json
  {
    "image_url": ["https://example.com/product-image.jpg","https://example.com/product-image2.jpg"]
  }
  ```
- **Response:**

```json
{
  "colors": ["#4s4d6s", "#0c0c0c"]
}
```
### 2️⃣ **Generate Product Name**
- **Endpoint:** `POST /generate-product-name`
- **Description:** `Generates a relevant product name based on the extracted colors and image content.`
- **Request:**
```json
{
  "image_url": ["https://example.com/product-image.jpg"]
}
```
- **Response:**
```json
{
  "product_name": "[Your brand name] Elegant Red & Blue Sneakers"
}
```
### 3️⃣ **Generate Detailed Product Description**
- **Endpoint:** `POST /generate-product-description`
- **Description:** `Creates a detailed and engaging product description based on the image and extracted features.`
- **Request:**
```json
{
  "image_url": ["https://example.com/product-image.jpg"],
  "product_name": "Elegant Red & Blue Sneakers",
  "colors": ["#4s4d6s"]
}
```
- **Response:**
```json
{
  "description": "Step out in style with these Elegant Red & Blue Sneakers. Designed for comfort and durability, they feature a lightweight build and a trendy design. Perfect for casual wear or sports activities!"
}
```
### 🛠 **Setup & Installation**
1.**Clone the Repository**
```sh
git clone https://github.com/Abdallah035/Motager-AI-Product-Helper.git
cd Motager-AI-Product-Helper
```
2.**Install Dependencies**
```sh
pip install -r requirements.txt
```
3.**🔑 Setting Up API Key**
    ***1. Create a `.env` file in the root directory.***
    ***2. Add your API key in this format:***
        API_KEY=your_api_key_here
        
4.**Run the FastAPI Server**
```sh
uvicorn main:app --reload
```
5.**Access API Documentation**
Open your browser and go to:
**📌 Swagger UI: http://127.0.0.1:8000/docs**

**📌 Redoc UI: http://127.0.0.1:8000/redoc**

### 🏆 **Why Use Motager AI Product Generator?**
**✔ Saves Time: Automates product listing and content generation.**

**✔ Improves Accuracy: AI ensures relevant and engaging descriptions.**

**✔ Easy Integration: FastAPI backend allows seamless API access.**
