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
import dropbox
from dropbox.exceptions import ApiError
import uuid

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")

# Initialize Together AI client
client = Together(api_key=TOGETHER_API_KEY)

# Initialize Dropbox client
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

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
    kernel = np.ones((2,2), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, kernel_sharpening)
    return Image.fromarray(gray)

# Upload image to Dropbox and create a shared link
def upload_to_dropbox(image_bytes):
    try:
        # Generate a unique filename
        unique_filename = f"/invoice_{uuid.uuid4().hex}.png"
        
        # Upload file to Dropbox
        dbx.files_upload(image_bytes.getvalue(), unique_filename, mode=dropbox.files.WriteMode.overwrite)
        
        # Create a shared link with direct download instead of preview page
        shared_link = dbx.sharing_create_shared_link_with_settings(
            unique_filename,
            dropbox.sharing.SharedLinkSettings(
                requested_visibility=dropbox.sharing.RequestedVisibility.public
            )
        )
        
        # Convert the standard shared link to a direct link
        # Format: https://www.dropbox.com/s/abc123/file.png?dl=0
        # Needs to become: https://dl.dropboxusercontent.com/s/abc123/file.png
        url = shared_link.url
        if '?dl=0' in url:
            url = url.replace('?dl=0', '')
        if 'www.dropbox.com' in url:
            url = url.replace('www.dropbox.com', 'dl.dropboxusercontent.com')
        
        logging.info(f"File uploaded to Dropbox: {url}")
        
        # Debug the URL
        try:
            test_response = requests.head(url)
            logging.info(f"Image URL status code: {test_response.status_code}")
        except Exception as e:
            logging.error(f"Error checking image URL: {e}")
            
        # Return both the URL and file path for later deletion
        return url, unique_filename
        
    except ApiError as e:
        logging.error(f"Dropbox API error: {e}")
        raise Exception(f"Dropbox upload failed: {e}")

# Cleanup function to delete file after processing
def cleanup_dropbox_file(file_path):
    try:
        dbx.files_delete_v2(file_path)
        logging.info(f"Successfully deleted file: {file_path}")
    except Exception as e:
        logging.error(f"Error deleting file: {e}")

def image_to_ocr(image):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # 'en' for English

    results = ocr.ocr(np.array(image), cls=True)
    output = ""

    for line in results[0]:
        points, (text, confidence) = line
        x1, y1 = points[0]
        output = output + text + " ( x = " + str(x1) + ")\n" 

    return output 

def contains_json(text):
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False

def extract_first_json(text):
    try:
        # If the whole text is valid JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON pattern in the text
        try:
            # Look for JSON between curly braces
            matches = re.findall(r'\{.*?\}', text, re.DOTALL)
            for match in matches:
                try:
                    result = json.loads(match)
                    # Make sure it has expected structure
                    if isinstance(result, dict) and any(key in result for key in ["medicines", "invoice_date", "other_dates", "extracted_text"]):
                        return result
                except json.JSONDecodeError:
                    continue
                
            # More aggressive JSON extraction for nested or malformed JSON
            potential_json = re.search(r'\{\s*"medicines"\s*:.*?"extracted_text"\s*:.*?\}', text, re.DOTALL)
            if potential_json:
                # Clean up potential issues
                clean_json = re.sub(r',\s*}', '}', potential_json.group())
                clean_json = re.sub(r',\s*,', ',', clean_json)
                try:
                    return json.loads(clean_json)
                except:
                    pass
        except:
            pass
    
    # Last resort - try to manually extract information
    try:
        result = {"medicines": [], "invoice_date": None, "other_dates": [], "extracted_text": text}
        
        # Try to find dates in common formats
        date_patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # DD/MM/YYYY or DD-MM-YYYY
            r'Date\s*:\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # Date: DD/MM/YYYY
            r'Invoice Date\s*:\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})'  # Invoice Date: DD/MM/YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Format as DD-MM-YYYY
                day, month, year = matches[0]
                if len(year) == 2:
                    year = "20" + year
                result["invoice_date"] = f"{day.zfill(2)}-{month.zfill(2)}-{year}"
                break
                
        return result
    except:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    file_path = None  # Initialize file path for cleanup
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        file_bytes = np.frombuffer(image_file.read(), np.uint8)

        # Process the image for better OCR results
        processed_image = image_process(file_bytes)
        processed_image.save("output.png")

        # Get comprehensive text using both processed and original images
        text = image_to_ocr(processed_image)

        # Converting processed image to bytes for uploading
        image_bytes = io.BytesIO()
        processed_image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        # Upload to Dropbox for LLM to access
        uploaded_image_url, file_path = upload_to_dropbox(image_bytes)

        retry = 0
        best_response = None
        best_json = None
        best_score = -1
        
        # Try with progressively different parameters
        temperatures = [0.2, 0.5, 0.7, 0.9]

        while(retry < maxretry):
            # Vary temperature based on retry count
            current_temp = temperatures[retry % len(temperatures)]
            
            prompt = (
                """
                You are an expert invoice parser specializing in pharmaceutical and medical invoices.

                Given the text extracted from an invoice image using PaddleOCR and with position of coordinates, And the image itself, carefully identify:

                1. MEDICINES/PRODUCTS: Look for items listed in columns labeled "Item", "Product", "Product Name", "Medicine Name", "Description", or similar. Sometimes products are listed in table format without explicit headers.

                2. DATES:
                   - Invoice date: Usually labeled as "Date", "Invoice Date", etc. (format as DD-MM-YYYY)
                   - Other dates: Extract all other dates found on the invoice (e.g., expiry, manufacturing) (format as MM-YY, DD-MM-YYYY, MM/YY, MM/YYYY, DD/MM/YYYY or similar)
    
                For any date in MM/YY format, convert to 01-MM-20YY.
                
                IMPORTANT INSTRUCTIONS:
                - Focus on medicine names in product lists, not on header/footer company information
                - Use visual layout from the image to identify table structures that might contain products
                - Check for items with quantities, batch numbers, or prices as these are likely products
                - Look for all dates in the invoice without trying to categorize them (except invoice date)
                - If information appears in tables, use column headers to identify data
                - Be thorough in scanning the full document
                
                Input Text:
                """
                + "\n" + text +"\n" + 
                """
                Also analyze the provided image for additional context and information not captured by OCR.

                Respond ONLY with a valid JSON object in the following format:
                {
                  "medicines": [
                    "Full Product Name 1",
                    "Full Product Name 2",
                    ...
                  ],
                  "invoice_date": "DD-MM-YYYY",
                  "other_dates": [
                    "DD-MM-YYYY",
                    ...
                  ],
                  "extracted_text": "full text from invoice"
                }

                """
                + ("ONLY JSON OUTPUT " * retry) + 

                """
                If there are no medicines mentioned or no dates found, include empty arrays or null values in the corresponding fields.
                """
            )
            
            logging.info(f"Making API call with image URL: {uploaded_image_url}")
            
            try:
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
                    temperature=current_temp,
                    stop=["<|eot_id|>", "<|eom_id|>"]
                )

                response_text = response.choices[0].message.content if response.choices else ""
                logging.info(f"LLM Response:\n{response_text}")
                print(f"LLM Response (try {retry+1}):\n{response_text[:500]}...")

                # Try to extract JSON from response
                extracted_json = extract_first_json(response_text)
                
                # If we found valid JSON
                if extracted_json:
                    # Always preserve the original OCR text, don't let the LLM replace it
                    extracted_json['extracted_text'] = text
                    
                    # Ensure all expected fields are present and transform old format to new format if needed
                    if 'medicines' not in extracted_json:
                        extracted_json['medicines'] = []
                    
                    if 'invoice_date' not in extracted_json:
                        extracted_json['invoice_date'] = None
                    
                    # Handle the case where we get old format JSON with expiry_dates and manufacturing_dates
                    if 'other_dates' not in extracted_json:
                        extracted_json['other_dates'] = []
                        
                        # Merge expiry_dates and manufacturing_dates if they exist in the old format
                        if 'expiry_dates' in extracted_json:
                            extracted_json['other_dates'].extend(extracted_json.pop('expiry_dates', []))
                        
                        if 'manufacturing_dates' in extracted_json:
                            extracted_json['other_dates'].extend(extracted_json.pop('manufacturing_dates', []))
                    
                    # Remove duplicate dates
                    if extracted_json['other_dates']:
                        extracted_json['other_dates'] = list(set(extracted_json['other_dates']))
                    
                    # Score the response - more complete responses score higher
                    score = 0
                    if extracted_json.get('medicines', []):
                        score += len(extracted_json['medicines']) * 2
                    if extracted_json.get('invoice_date'):
                        score += 5
                    if extracted_json.get('other_dates', []):
                        score += len(extracted_json['other_dates'])
                    
                    # Track best result
                    if best_json is None or score > best_score:
                        best_json = extracted_json
                        best_response = response_text
                        best_score = score
        
                    # Return immediately if we have a high-quality result
                    if score >= 8:
                        logging.info(f"Found high-quality result with score {score}, returning immediately")
                        result = jsonify(best_json)
                        
                        # Clean up the file before returning
                        if file_path:
                            cleanup_dropbox_file(file_path)
                            
                        return result
                
            except Exception as api_error:
                logging.error(f"API error on attempt {retry+1}: {api_error}")
                print(f"API error on attempt {retry+1}: {api_error}")
            
            retry = retry + 1
            logging.info(f"Retrying {retry}/{maxretry}, current score: {best_score}")
            print(f"Retrying {retry}/{maxretry}, current score: {best_score}")

        # When we reach max retries, return the best result we have
        if best_json:
            output = {
                "extracted_text": text,
                "best_json": best_json
            }
            logging.info(f"Returning best result with score {best_score} after {retry} tries")
            result = jsonify(output)
        else:
            logging.info("No valid JSON found after all retries, returning empty result")
            result = jsonify({
                "medicines": [],
                "invoice_date": None,
                "other_dates": [],
                "extracted_text": text
            })
            
        # Clean up the file before returning the result
        if file_path:
            cleanup_dropbox_file(file_path)
            
        return result

    except Exception as e:
        logging.exception("Error in processing image")
        
        # Make sure to clean up even if we encounter an error
        if file_path:
            try:
                cleanup_dropbox_file(file_path)
            except Exception as cleanup_error:
                logging.error(f"Failed to clean up file after error: {cleanup_error}")
                
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
