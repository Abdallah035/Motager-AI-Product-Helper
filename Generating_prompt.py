import time
from concurrent.futures import ThreadPoolExecutor
from Generate_caption import generate_caption
from Color_extraction import extract_colors
from soupsieve.util import lower


def generate_product_name_prompt(image_path,Brand_name,vgg16_model,model,tokenizer):
    caption = generate_caption(image_path[0],vgg16_model,model,tokenizer)
    if (str(Brand_name).lower()) == "none" or str(Brand_name).lower()=="generic":
        statment = (f"Generate a product title name based on the caption {caption} as following information."
                )
    else:
        statment = (f"Generate a product title name based on the caption {caption} and {Brand_name} as following information."
                    f"Replace the brand name with {Brand_name}. if it is not 'none' or 'None' or 'Generic'"
                    f"remember if {Brand_name} is 'none' or 'None' or 'Generic' do not depend on it and exclude it from product name"
                    f" Ensure the product title follows this format:<{Brand_name}>  <Product Details>. "
                    )
    prompt = (f"{statment}"
              f"reformat the {caption} to be professional product title like in the Amazon for website"
              f"Ensure that is only one product title"
              f"The product details should include features like product type and it must be somthing popular, series name, purpose "
              f"and any relevant specifics"
              f"Do NOT use escape characters or newline (\n)."
              f"excluding those words (startseq) and (endseq) removing any extra spaces."
              f"excluding any color and brand name from the product title without any(:) and (,)."
              f"example: 'Adidas T-Shirts Round Neck Cotton Full Sleeve'"
              f"do not say specific model number for the product title."
       
              f"example if brand name is Apple and caption is smartphone provide that it is iphone but do not provide it's model number as (15 pro max) "
              f"examples: iphone [no] pro max , Samsung S[no] ultra , Samsung A[no] get the model but not be very specific"
              f"do not generate none or generic in product name"

                  )
    return prompt


def generate_description_prompt(product_name, vgg16_model, model, tokenizer):
    prompt = (
        f'Generate a product description with the following sections: "About this item" and "Product description".'
        f'Based on this information:'
        f'Product Title: {product_name}'
        f'Important Requirements:'
        f'1. Limit the description to exactly 150 words.'
        f'2. Extract the brand name from the Product Title below and use it to reference the product within the description.'
        f'3. Follow the structure provided below for "About this item" and "Product description".'
        f'4. Ensure each line in the description contains two sentences, removing unnecessary spaces after periods (.).'
        f'5. Do NOT use escape characters or newline (\n).'

        f'Expected Output Format:'

        f'About this item Genuine leather construction for lasting durability.Multiple card slots and compartments for organization.Sleek and sophisticated design for a polished look.Compact size for easy carrying in pockets or bags.Secure closure to protect your valuables.Product Description.The polo leather wallet offers a premium feel and functionality. It\'s crafted from high-quality leather, ensuring both style and longevity.Its thoughtful design includes ample space for cards and cash. The compact size makes it ideal for everyday use.This polo leather wallet is a perfect blend of practicality and sophistication. It’s designed for the modern gentleman who appreciates quality.'
        f'Remember to:'
        f'remove any ("\n") in response'
        f'Each bullet in "About this item" should only have a maximum of 6 words.'
        f'Ensure each line in the description contains two sentences.'
        f'Remove and exclude extra spaces after (.).'
    )

    return prompt



