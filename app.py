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
import cloudinary
import cloudinary.uploader
import cloudinary.api
import uuid

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Initialize Together AI client
client = Together(api_key=TOGETHER_API_KEY)

# Initialize Cloudinary
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True
)

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

def compress_image(image_bytes, max_size_bytes=9*1024*1024, initial_quality=85):
    """Compress image to ensure it's under the Cloudinary size limit"""
    # Convert bytes to PIL Image
    img = Image.open(image_bytes)
    
    # Start with a buffer to store compressed image
    output_buffer = io.BytesIO()
    
    # Try with initial quality
    quality = initial_quality
    img.save(output_buffer, format='JPEG', quality=quality, optimize=True)
    
    # Continue reducing quality until we're under the limit
    while output_buffer.tell() > max_size_bytes and quality > 20:
        # Reset buffer and reduce quality
        output_buffer = io.BytesIO()
        quality -= 10
        logging.info(f"Compressing image with quality: {quality}")
        img.save(output_buffer, format='JPEG', quality=quality, optimize=True)
    
    # If still too large, resize the image
    if output_buffer.tell() > max_size_bytes:
        # Calculate new dimensions to reduce size
        original_width, original_height = img.size
        scale_factor = 0.8  # Reduce to 80% of size
        
        while output_buffer.tell() > max_size_bytes and scale_factor > 0.3:
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Reset buffer and save resized image
            output_buffer = io.BytesIO()
            resized_img.save(output_buffer, format='JPEG', quality=quality, optimize=True)
            
            # Reduce scale factor for next iteration if needed
            scale_factor -= 0.1
            logging.info(f"Resized image to {new_width}x{new_height} with quality {quality}")
    
    # Reset pointer to beginning of buffer
    output_buffer.seek(0)
    
    final_size_mb = output_buffer.tell() / (1024 * 1024)
    logging.info(f"Final compressed image size: {final_size_mb:.2f} MB")
    
    return output_buffer

def upload_to_cloudinary(image_bytes):
    try:
        # Generate a unique filename
        unique_id = uuid.uuid4().hex
        
        # Compress the image before uploading to stay under 10MB limit
        compressed_image = compress_image(image_bytes)
        
        # Upload file to Cloudinary
        upload_result = cloudinary.uploader.upload(
            compressed_image.getvalue(),
            public_id=f"invoice_{unique_id}",
            folder="invoice_parser",
            resource_type="image",
            # Add these parameters to ensure proper URL access
            access_mode="public",
            type="upload"
        )
        
        # Get the secure URL for the uploaded image
        url = upload_result['secure_url']
        public_id = upload_result['public_id']
        
        logging.info(f"File uploaded to Cloudinary with URL: {url}")
        return url, public_id
        
    except Exception as e:
        logging.error(f"Cloudinary upload error: {e}")
        raise Exception(f"Cloudinary upload failed: {e}")

# Add a new function to handle image download and resizing
def prepare_image_for_api(image_url, file_path=None):
    """Download image from URL, resize if needed, and return a base64 encoded version"""
    try:
        # Try to download the image
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            logging.error(f"Failed to download image: {response.status_code}")
            # Fall back to local file
            with open("output.png", "rb") as f:
                img_data = f.read()
        else:
            img_data = response.content
            
        # Load image and resize if too large
        image = Image.open(io.BytesIO(img_data))
        
        # Resize if the image is too large (keep aspect ratio)
        max_size = 1024  # Maximum dimension
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            logging.info(f"Resized image from {image.size} to {new_size}")
            
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Convert to base64
        import base64
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
        
    except Exception as e:
        logging.error(f"Error preparing image: {e}")
        # Fall back to local file as base64
        try:
            with open("output.png", "rb") as img_file:
                import base64
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                return f"data:image/png;base64,{img_data}"
        except Exception as e2:
            logging.error(f"Error with fallback image: {e2}")
            raise Exception(f"Failed to prepare image: {e}")
                
# Cleanup function to delete file after processing
def cleanup_cloudinary_file(public_id):
    try:
        cloudinary.uploader.destroy(public_id)
        logging.info(f"Successfully deleted file: {public_id}")
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
    public_id = None  # Initialize public_id for cleanup
    
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
        processed_image.save(image_bytes, format="JPEG", quality=90)
        image_bytes.seek(0)

        # Upload to Cloudinary for LLM to access
        uploaded_image_url, public_id = upload_to_cloudinary(image_bytes)

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
                # Verify image URL is accessible
                try:
                    response = requests.head(uploaded_image_url)
                    if response.status_code != 200 or not response.headers.get('Content-Type', '').startswith('image/'):
                        logging.error(f"Image URL is not accessible or not an image: {uploaded_image_url}")
                        logging.error(f"Content-Type: {response.headers.get('Content-Type', 'unknown')}")
                        
                        # Fall back to base64
                        logging.info("Preparing base64 encoded image")
                        with open("output.png", "rb") as img_file:
                            import base64
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                            img_url = f"data:image/png;base64,{img_data}"
                            logging.info("Using base64 encoded image")
                    else:
                        img_url = uploaded_image_url
                        logging.info(f"Using Cloudinary URL: {img_url}")
                        
                    # Properly format the API request
                    response = client.chat.completions.create(
                        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": img_url}}
                                ]
                            }
                        ],
                        temperature=current_temp,
                        max_tokens=4096,
                    )
                
                except requests.exceptions.RequestException as url_error:
                    logging.error(f"Error accessing image URL: {url_error}")
                    # Fall back to base64 if URL access fails
                    logging.info("Falling back to base64 encoded image due to URL access failure")
                    with open("output.png", "rb") as img_file:
                        import base64
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        img_url = f"data:image/png;base64,{img_data}"
                    
                    # Try API call with base64 image
                    response = client.chat.completions.create(
                        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": img_url}}
                                ]
                            }
                        ],
                        temperature=current_temp,
                        max_tokens=4096,
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
                        if public_id:
                            cleanup_cloudinary_file(public_id)
                            
                        return result
                
            except Exception as api_error:
                logging.error(f"API error on attempt {retry+1}: {api_error}")
                print(f"API error on attempt {retry+1}: {api_error}")
            
            retry = retry + 1
            logging.info(f"Retrying {retry}/{maxretry}, current score: {best_score}")
            print(f"Retrying {retry}/{maxretry}, current score: {best_score}")

        # When we reach max retries, return the best result we have
        if best_json:
            # Return just the best JSON instead of wrapping it
            result = jsonify(best_json)
        else:
            logging.info("No valid JSON found after all retries, returning empty result")
            result = jsonify({
                "medicines": [],
                "invoice_date": None,
                "other_dates": [],
                "extracted_text": text
            })
            
        # Clean up the file before returning the result
        if public_id:
            cleanup_cloudinary_file(public_id)
            
        return result

    except Exception as e:
        logging.exception("Error in processing image")
        
        # Make sure to clean up even if we encounter an error
        if public_id:
            try:
                cleanup_cloudinary_file(public_id)
            except Exception as cleanup_error:
                logging.error(f"Failed to clean up file after error: {cleanup_error}")
                
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
