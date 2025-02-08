from Color_extraction import extract_colors
from Generate_productName_description import generate_product_name, generate_description
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not set. Please configure your .env file or system environment.")

image_path_list = ['images_test/test/27.jpg']
# product_name = generate_product_name(image_path_list)
# print(product_name)
# color_list = extract_colors(image_path_list)
description = generate_description(image_path_list,API_KEY,"iphone 16 pro max",color_list=["#be7c60"])
print(description)