from Color_extraction import extract_colors
# from Generate_productName_description import generate_product_name, generate_description
from dotenv import load_dotenv
import os
from Generate_caption import extract_image_features_one
# from Generate_productName_description import clean_response
# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not set. Please configure your .env file or system environment.")

# image_path_list = ['https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRcbfffkBR71xadfZ38APy1tclW2zQ77c6--g&s']
#
# product_name = generate_product_name(image_path_list,'Samsung',API_KEY)
# print(product_name)
# text = "None"
# print((text.lower()))
# color_list = extract_colors(image_path_list)
# print(color_list)
# description = generate_description(image_path_list,API_KEY,product_name,color_list)
# print(description)
# image = url_to_cv2_image("https://duuw10jl1n.ufs.sh/f/URa8oGmtpSmeY9aosOAeRgyf9hO1udBMVQv2tTG7YlCD8XLi")
# print(image)