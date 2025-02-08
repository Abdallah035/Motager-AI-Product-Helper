from Generating_prompt import generate_product_name_prompt , generate_description_prompt
import google.generativeai as genai
image_path_list = ['images_test/test/64.jpg']


def generate_product_name(image_path_list,api_key):
    prompt = generate_product_name_prompt(image_path_list)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


def generate_description(image_path_list,api_key,product_name,color_list=None):
    prompt = generate_description_prompt(image_path_list,product_name,color_list=color_list)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

