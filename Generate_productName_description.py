from Generating_prompt import generate_product_name_prompt , generate_description_prompt
import google.generativeai as genai

def clean_response(text: str) -> str:
    return text.replace("\n", " ").strip()


def generate_product_name(image_path_list, Brand_name, vgg16_model, model, tokenizer, api_key):
    prompt = generate_product_name_prompt(image_path_list, Brand_name, vgg16_model, model, tokenizer)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return str(response.text) if response.text else ""


def generate_description(api_key, product_name, vgg16_model, model, tokenizer):
    prompt = generate_description_prompt(product_name, vgg16_model, model, tokenizer)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return str(response.text) if response.text else ""

