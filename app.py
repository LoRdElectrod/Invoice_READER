from flask import Flask, request, jsonify, render_template
import os
import requests
import logging
from together import Together
from dotenv import load_dotenv
import re
import json

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")

# Initialize Together AI client
client = Together(api_key=TOGETHER_API_KEY)

app = Flask(__name__, template_folder="templates")

# Enable logging
logging.basicConfig(level=logging.INFO)

# Upload image to Imgur
def upload_to_imgur(image_path):
    headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
    with open(image_path, "rb") as image_file:
        response = requests.post("https://api.imgur.com/3/upload", headers=headers, files={"image": image_file})
    if response.status_code == 200:
        return response.json()["data"]["link"]
    else:
        raise Exception(f"Imgur upload failed: {response.json()}")

# Extract dates from text using regex
def extract_dates(text):
    date_patterns = [
        r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b',
        r'\b(\d{1,2})[/-](\d{4})\b',
        r'\b(\d{4})[/-](\d{1,2})\b',
    ]
    found_dates = []

    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) == 3:
                dd, mm, yyyy = match
            elif len(match) == 2 and len(match[1]) == 4:  # mm-yyyy
                dd, mm, yyyy = "01", match[0], match[1]
            elif len(match[0]) == 4:  # yyyy-mm
                dd, mm, yyyy = "01", match[1], match[0]
            else:
                continue

            dd = dd.zfill(2)
            mm = mm.zfill(2)
            yyyy = yyyy if len(yyyy) == 4 else "20" + yyyy
            formatted_date = f"{dd}-{mm}-{yyyy}"
            if formatted_date not in found_dates:
                found_dates.append(formatted_date)

    return found_dates

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        image_path = f"./temp/{image_file.filename}"
        os.makedirs("./temp", exist_ok=True)
        image_file.save(image_path)

        uploaded_image_url = upload_to_imgur(image_path)

        prompt = (
            "Extract the full raw text from this invoice image first.\n\n"
            "Then, identify all product or medicine names listed under the items/products section, "
            "and return ONLY their names in a separate section as a JSON array like this:\n\n"
            "[ { \"product\": \"Product Name 1\" }, { \"product\": \"Product Name 2\" } ]\n\n"
            "Only include actual item names. Do NOT include prices, quantities, batch numbers, etc.\n\n"
            "Format your response in two parts:\n\n"
            "1. Extracted Text:\n<Full invoice text here>\n\n"
            "2. Product Names (JSON):\n[ { \"product\": \"Product 1\" }, { \"product\": \"Product 2\" } ]"
        )

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": uploaded_image_url}}
                    ]
                }
            ],
            max_tokens=None,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"]
        )

        response_text = response.choices[0].message.content if response.choices else ""
        logging.info(f"LLM Response:\n{response_text}")

        # Initialize fallback values
        extracted_text = ""
        product_names = []

        # Try multiple ways to parse
        patterns = [
            r"\*\*Extracted Text:\*\*(.*?)\*\*Product Names \(JSON\):\*\*(.*)",
            r"1\. Extracted Text:(.*?)2\. Product Names \(JSON\):(.*)"
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                extracted_text = match.group(1).strip()
                product_json = json.loads(match.group(2).strip())
                product_names = [item["product"] for item in product_json if "product" in item]
                break
        else:
            logging.warning("Expected pattern not found in LLM output.")

        extracted_dates = extract_dates(extracted_text)

        return jsonify({
            "extracted_text": extracted_text,
            "product_names": product_names,
            "dates": extracted_dates
        })

    except Exception as e:
        logging.exception("Error in processing image")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run()
