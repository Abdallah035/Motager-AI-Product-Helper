import time
from concurrent.futures import ThreadPoolExecutor
from Generate_caption import generate_caption
from Color_extraction import extract_colors

def generate_product_name_prompt(image_path):
    caption = generate_caption(image_path[0])
    prompt = (f"Generate a product title name based on the caption {caption} and following information. "
              f"Replace the brand name with [your brand name]."
              f" Ensure the product title follows this format: [your brand name]  <Product Details>. "
              f"The product details should include features like product type and it must be somthing popular, series name, purpose "
              f"and any relevant specifics"
              f"excluding those words (startseq) and (endseq) removing any extra spaces."
              f"excluding any color and brand name from the product title without any(:) and (,)."
              f"example: '[your brand name]T-Shirts Round Neck Cotton Full Sleeve'"
              )
    return prompt

def generate_description_prompt(image_path, product_name, color_list=None):
    caption = generate_caption(image_path[0])  # Get caption from the image

    # Determine the colors line if colors are provided
    if color_list:
        color_statement = (f"Include the following color details: "
                           f"exclude any colors in caption {caption}"
                           f"Replace these hex codes {color_list} with color names. "
                           f"only use colors in {color_list} as avaliable colors"
                           f"Display them as: `<strong>ColorName</strong>"
                           f"at the final line without additional sentences.")
    else:
        color_statement = "No colors provided. Focus on materials, fit, and benefits without using colors."

    # Generate the prompt
    colors_line = (
        f"Available colors: "
        + ", ".join(f"<strong>{color.upper()}</strong>" for color in color_list)
        + ".</p>"
        if color_list
        else ""
    )

    prompt = (
        f'Generate a product description with the following sections: "About this item" and "Product description".\n\n'
        f'based on this information:'
        f'Caption: {caption}\n'
        f'Product Title: {product_name}\n'
        f'{color_statement}\n\n'
        f'Important Requirements:\n'
        f'1. Limit the description to exactly 150 words.\n'
        f'2. Extract the brand name from the Product Title below and use it to reference the product within the description.\n'
        f'3. Do not include brand details from the Caption below.\n'
        f'4. Exclude the words (startseq) and (endseq) from the Caption.\n'
        f'5. Follow the structure provided below for "About this item" and "Product description".\n'
        f'6. Ensure each line in the description contains two sentences, removing unnecessary spaces after periods (.).\n'
        f'7. If colors are provided, include them as the last line in the description and format them using HTML `<strong>` tags.\n'

        f'Expected Output Format:\n\n'

        f'About this item\n\n '
        f'. Genuine leather construction for lasting durability.\n'
        f'. Multiple card slots and compartments for organization.\n'
        f'. Sleek and sophisticated design for a polished look.\n'
        f'. Compact size for easy carrying in pockets or bags.\n'
        f'. Secure closure to protect your valuables.\n'
        f'Product description\n\n'
        f"The polo leather wallet offers a premium feel and functionality.It's crafted from high-quality leather, ensuring both style and longevity.\n"
        f'Its thoughtful design includes ample space for cards and cash. The compact size makes it ideal for everyday use.\n'
        f'This polo leather wallet is a perfect blend of practicality and sophistication. It’s designed for the modern gentleman who appreciates quality. \n'
        f'Available colors:  {colors_line}\n' 

        f'Remember to:\n'
        f'each bullet in about this item should only have at maximum 6 words'
        f'Ensure each line in the description contains two sentences'
        f'removing and excluding extra spaces after (.)'
        f"- Place the color line at the end of the description like that  'Available colors: red ' \n"
    )

    return prompt
