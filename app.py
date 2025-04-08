from flask import Flask, request, jsonify, render_template
import os
import requests
import logging
from together import Together
from dotenv import load_dotenv
import re
import json
import cv2
from PIL import Image 
import io
from paddleocr import PaddleOCR
import numpy as np

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")

# Initialize Together AI client
client = Together(api_key=TOGETHER_API_KEY)

app = Flask(__name__, template_folder="templates")

# Enable logging
logging.basicConfig(level=logging.INFO)

# Retry if output is not JSON
maxretry = 10

# Process Image before applying OCR
def image_process(file_bytes):
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 10, 15, 20)
    # gray = cv2.equalizeHist(gray)
    return Image.fromarray(gray)

# Upload image to Imgur
def upload_to_imgur(image_bytes):
    headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
    # with open(image_path, "rb") as image_file:
    response = requests.post("https://api.imgur.com/3/upload", headers=headers, files={"image": image_bytes})
    if response.status_code == 200:
        return response.json()["data"]["link"]
    else:
        raise Exception(f"Imgur upload failed: {response.json()}")

# Extract dates from text using regex



def image_to_ocr(image):

    image.show()
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # 'en' for English

    results = ocr.ocr(np.array(image), cls=True)
    text = ""

    for line in results[0]:
      text = text + line[1][0] + "\n"
    return text


def extract_dates(text):
    found_dates = []

    patterns = [
        r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b',  # DD/MM/YYYY or D-M-YY
        r'\b(\d{1,2})[/-](\d{2})\b',                 # MM/YY
        r'\b(\d{4})[/-](\d{1,2})\b'                  # YYYY/MM
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            parts = match.groups()

            try:
                if len(parts) == 3:
                    dd, mm, yyyy = parts
                    if int(dd) > 31:
                        dd, mm, yyyy = "01", mm, dd  # Maybe reversed
                    if len(yyyy) == 2:
                        yyyy = "20" + yyyy
                    dd = dd.zfill(2)
                    mm = mm.zfill(2)

                elif len(parts) == 2:
                    mm, yy = parts
                    dd = "01"
                    mm = mm.zfill(2)
                    yyyy = "20" + yy

                else:
                    continue

                # Basic sanity checks
                if not (1 <= int(mm) <= 12):
                    continue
                if not (1900 <= int(yyyy) <= 2100):
                    continue

                formatted = f"{dd}-{mm}-{yyyy}"
                if formatted not in found_dates:
                    found_dates.append(formatted)

            except Exception:
                continue  # Skip invalid matches

    return found_dates



def contains_json(text):
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False

def extract_first_json(text):
    matches = re.findall(r'\{.*?\}', text)
    for match in matches:
        try:
            return json.loads(match)  
        except json.JSONDecodeError:
            continue
    return None  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        file_bytes = np.frombuffer(image_file.read(), np.uint8)

        image = image_process(file_bytes)

        # Converting processed image to bytes for uploading
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        text = image_to_ocr(image)        

        uploaded_image_url = upload_to_imgur(image_bytes)

        retry = 0

        while(retry < maxretry):

            prompt = (
                """
                You are an intelligent OCR post-processor.

                Given a list of text blocks extracted from an invoice, identify and extract a list of medicines.

                Input:
                """
                + "\n" + text +"\n" + 
                """
                Also use Provided image as context.
                Check columns in image with "Item", "Product", "Product Name", "Medicine Name", "Description".

                
                Do not include markdown, do not wrap with triple backticks, no explanations,
                Respond ONLY with a JSON object in the following format:
                {
                  "medicines": [
                    "name",
                  ],
                  "invoice_date": "date"
                }

                """
                + ("Only JSON " * retry) + 

                """
                If there are No Medicines mentioned, just response with empty JSON, like this:
                {}
                """
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
            print(response_text)


            if contains_json(response_text) is not True:
                retry = retry + 1
                logging.info(f"Retrying {retry}\n")
                print("retrying "+str(retry))
                continue

            extracted_json = json.loads(response_text);

            return jsonify({
               "extracted_text": extracted_json,
            })

            break

        return jsonify({
            "extracted_text": "",
        })


    except Exception as e:
        logging.exception("Error in processing image")
        # return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run()
    # process_image()



