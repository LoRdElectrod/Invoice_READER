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
import numpy as np
import cloudinary
import cloudinary.uploader
import cloudinary.api
import uuid
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
from collections import defaultdict
import easyocr  # Alternative lightweight OCR as fallback

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

# Initialize LayoutLMv3 - Primary OCR and structure extraction
try:
    layoutlm_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-large")
    layoutlm_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-large")
    logging.info("LayoutLMv3 model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load LayoutLMv3: {e}")
    layoutlm_processor = None
    layoutlm_model = None

# Initialize EasyOCR as lightweight fallback
try:
    easyocr_reader = easyocr.Reader(['en'])
    logging.info("EasyOCR initialized as fallback")
except Exception as e:
    logging.warning(f"EasyOCR initialization failed: {e}")
    easyocr_reader = None

# Process Image before applying OCR
def image_process(file_bytes):
    """Enhanced image preprocessing for better OCR results"""
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize for better OCR (increase resolution)
    height, width = gray.shape
    if width < 1000:  # Only upscale if image is small
        scale_factor = 1000 / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Noise reduction
    gray = cv2.medianBlur(gray, 3)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Sharpening
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, kernel_sharpening)
    
    # Convert back to PIL Image
    return Image.fromarray(gray)

def compress_image(image_bytes, max_size_bytes=9*1024*1024, initial_quality=85):
    """Compress image to ensure it's under the Cloudinary size limit"""
    img = Image.open(image_bytes)
    
    output_buffer = io.BytesIO()
    quality = initial_quality
    img.save(output_buffer, format='JPEG', quality=quality, optimize=True)
    
    while output_buffer.tell() > max_size_bytes and quality > 20:
        output_buffer = io.BytesIO()
        quality -= 10
        logging.info(f"Compressing image with quality: {quality}")
        img.save(output_buffer, format='JPEG', quality=quality, optimize=True)
    
    if output_buffer.tell() > max_size_bytes:
        original_width, original_height = img.size
        scale_factor = 0.8
        
        while output_buffer.tell() > max_size_bytes and scale_factor > 0.3:
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            output_buffer = io.BytesIO()
            resized_img.save(output_buffer, format='JPEG', quality=quality, optimize=True)
            
            scale_factor -= 0.1
            logging.info(f"Resized image to {new_width}x{new_height} with quality {quality}")
    
    output_buffer.seek(0)
    final_size_mb = output_buffer.tell() / (1024 * 1024)
    logging.info(f"Final compressed image size: {final_size_mb:.2f} MB")
    
    return output_buffer

def upload_to_cloudinary(image_bytes):
    try:
        unique_id = uuid.uuid4().hex
        compressed_image = compress_image(image_bytes)
        
        upload_result = cloudinary.uploader.upload(
            compressed_image.getvalue(),
            public_id=f"invoice_{unique_id}",
            folder="invoice_parser",
            resource_type="image",
            access_mode="public",
            type="upload"
        )
        
        url = upload_result['secure_url']
        public_id = upload_result['public_id']
        
        logging.info(f"File uploaded to Cloudinary with URL: {url}")
        return url, public_id
        
    except Exception as e:
        logging.error(f"Cloudinary upload error: {e}")
        raise Exception(f"Cloudinary upload failed: {e}")

def cleanup_cloudinary_file(public_id):
    try:
        cloudinary.uploader.destroy(public_id)
        logging.info(f"Successfully deleted file: {public_id}")
    except Exception as e:
        logging.error(f"Error deleting file: {e}")

def extract_with_layoutlmv3(image):
    """
    Primary extraction method using LayoutLMv3 for both OCR and structure detection
    """
    if not layoutlm_processor or not layoutlm_model:
        logging.error("LayoutLMv3 not available!")
        return None, None, None
    
    try:
        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # LayoutLMv3 can process image directly for OCR + structure
        encoding = layoutlm_processor(image, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = layoutlm_model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
        
        # Extract text and positions from the processor's tokenizer
        tokens = layoutlm_processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())
        
        # Get word-level information
        word_ids = encoding.word_ids()
        boxes = encoding["bbox"].squeeze().tolist() if "bbox" in encoding else []
        
        # Reconstruct text and positions
        ocr_data = []
        raw_text = ""
        current_word = ""
        current_box = None
        
        for i, (token, word_id) in enumerate(zip(tokens, word_ids)):
            if word_id is not None and token not in ['[CLS]', '[SEP]', '[PAD]']:
                if token.startswith('##'):
                    current_word += token[2:]  # Remove ## prefix
                else:
                    if current_word and current_box:
                        # Save previous word
                        ocr_data.append({
                            "text": current_word,
                            "confidence": 0.9,  # LayoutLMv3 doesn't provide confidence, assume high
                            "x1": current_box[0],
                            "y1": current_box[1], 
                            "x2": current_box[2],
                            "y2": current_box[3],
                            "x_mid": (current_box[0] + current_box[2]) / 2,
                            "y_mid": (current_box[1] + current_box[3]) / 2
                        })
                        raw_text += current_word + f" (x={current_box[0]}, y={current_box[1]})\n"
                    
                    current_word = token
                    current_box = boxes[i] if i < len(boxes) else [0, 0, 0, 0]
        
        # Don't forget the last word
        if current_word and current_box:
            ocr_data.append({
                "text": current_word,
                "confidence": 0.9,
                "x1": current_box[0],
                "y1": current_box[1],
                "x2": current_box[2], 
                "y2": current_box[3],
                "x_mid": (current_box[0] + current_box[2]) / 2,
                "y_mid": (current_box[1] + current_box[3]) / 2
            })
            raw_text += current_word + f" (x={current_box[0]}, y={current_box[1]})\n"
        
        # Organize by rows
        structured_rows = organize_by_rows(ocr_data)
        
        # Additional structure information from LayoutLMv3
        structure_info = {
            "predictions": predictions,
            "confidence_scores": [float(torch.softmax(outputs.logits, dim=-1).max(dim=-1)[0].mean())],
            "layout_detected": True
        }
        
        return raw_text, ocr_data, structured_rows, structure_info
        
    except Exception as e:
        logging.error(f"LayoutLMv3 extraction failed: {e}")
        return None, None, None, None
    
def fallback_ocr_extraction(image):
    """
    Fallback OCR using EasyOCR when LayoutLMv3 fails
    """
    if not easyocr_reader:
        logging.error("No OCR method available!")
        return "", [], []
    
    try:
        logging.info("Using EasyOCR fallback...")
        
        # Convert PIL image to numpy array for EasyOCR
        img_array = np.array(image)
        
        # EasyOCR extraction
        results = easyocr_reader.readtext(img_array)
        
        ocr_data = []
        raw_text = ""
        
        for detection in results:
            bbox, text, confidence = detection
            
            # Extract coordinates
            x1, y1 = bbox[0]
            x2, y2 = bbox[2]
            
            ocr_data.append({
                "text": text,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "x_mid": (x1 + x2) / 2,
                "y_mid": (y1 + y2) / 2
            })
            
            raw_text += text + f" (x={x1}, y={y1})\n"
        
        structured_rows = organize_by_rows(ocr_data)
        
        return raw_text, ocr_data, structured_rows
        
    except Exception as e:
        logging.error(f"Fallback OCR failed: {e}")
        return "", [], []

def organize_by_rows(ocr_data, y_tolerance=30):
    """Group OCR text items into rows based on y-coordinate proximity"""
    if not ocr_data:
        return []
    
    sorted_data = sorted(ocr_data, key=lambda x: x["y_mid"])
    rows = [[sorted_data[0]]]
    
    for item in sorted_data[1:]:
        last_row = rows[-1]
        last_row_y_avg = sum(i["y_mid"] for i in last_row) / len(last_row)
        
        if abs(item["y_mid"] - last_row_y_avg) <= y_tolerance:
            last_row.append(item)
        else:
            rows.append([item])
    
    # Sort items in each row by x-coordinate
    for row in rows:
        row.sort(key=lambda x: x["x_mid"])
    
    return rows

def advanced_table_extraction(structured_rows, structure_info=None):
    """Enhanced table extraction using coordinate analysis and LayoutLMv3 insights"""
    if not structured_rows:
        return []
    
    # Enhanced header detection
    header_keywords = {
        "product": ["item", "product", "medicine", "description", "name", "drug", "medication"],
        "batch": ["batch", "btch", "lot", "serial", "batch#", "lot#"],
        "expiry": ["exp", "expiry", "expire", "valid", "expiry date", "exp date"],
        "mfg": ["mfg", "manufacture", "made", "production", "mfg date"],
        "quantity": ["qty", "quantity", "units", "nos", "pieces", "pcs"],
        "price": ["price", "rate", "mrp", "amount", "cost", "value"],
        "date": ["date", "dt"]
    }
    
    column_mapping = {}
    best_header_row = None
    best_score = 0
    
    # Check first 3 rows for headers
    for row_idx, row in enumerate(structured_rows[:3]):
        score = 0
        temp_mapping = {}
        
        for col_idx, item in enumerate(row):
            text = item["text"].lower().strip()
            
            for field_type, keywords in header_keywords.items():
                if any(keyword in text for keyword in keywords):
                    temp_mapping[field_type] = col_idx
                    score += 3 if field_type == "product" else 1
        
        if score > best_score and "product" in temp_mapping:
            best_score = score
            best_header_row = row_idx
            column_mapping = temp_mapping
    
    # Remove header row from data rows
    data_rows = structured_rows.copy()
    if best_header_row is not None:
        data_rows = [row for i, row in enumerate(structured_rows) if i != best_header_row]
    
    # Extract structured products
    products = []
    
    for row in data_rows:
        if len(row) < 2:  # Skip rows with too few columns
            continue
            
        product_data = {}
        
        # Use column mapping if available
        if column_mapping:
            for field_type, col_idx in column_mapping.items():
                if col_idx < len(row):
                    value = row[col_idx]["text"].strip()
                    if value:
                        product_data[field_type] = value
        else:
            # Intelligent field detection based on content patterns
            product_name = None
            dates = []
            other_fields = {}
            
            for item in row:
                text = item["text"].strip()
                text_lower = text.lower()
                
                # Enhanced date pattern detection
                date_patterns = [
                    r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY
                    r'\d{1,2}[/-]\d{2,4}',  # MM/YY
                    r'[a-z]{3}\s*\d{2,4}',  # Mar 2024
                    r'\d{2,4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD
                ]
                
                is_date = any(re.search(pattern, text, re.IGNORECASE) for pattern in date_patterns)
                
                if is_date:
                    if any(kw in text_lower for kw in ["exp", "expiry", "expire"]):
                        other_fields["expiry_date"] = text
                    elif any(kw in text_lower for kw in ["mfg", "manufacture", "made"]):
                        other_fields["mfg_date"] = text
                    else:
                        dates.append(text)
                elif re.search(r'batch|btch|lot', text_lower):
                    other_fields["batch"] = text
                elif re.search(r'^\s*\d+(\.\d{1,2})?\s*$', text):
                    if "quantity" not in other_fields:
                        other_fields["quantity"] = text
                    elif "price" not in other_fields:
                        other_fields["price"] = text
                elif not product_name and len(text) > 2 and not any(kw in text_lower for kw in ["total", "amount", "qty", "page", "invoice"]):
                    product_name = text
            
            if product_name:
                product_data["product"] = product_name
                product_data.update(other_fields)
                if dates:
                    product_data["associated_dates"] = dates
        
        if product_data and ("product" in product_data or "medicine" in product_data):
            products.append(product_data)
    
    return products

def enhance_with_llm_correction(products, raw_text):
    """Use LLM to correct and enhance product names"""
    if not products:
        return products
    
    # Prepare product list for LLM correction
    product_names = []
    for product in products:
        name = product.get("product") or product.get("medicine") or "Unknown"
        product_names.append(name)
    
    correction_prompt = f"""
    You are a pharmaceutical expert. Please correct and standardize these medicine/product names extracted from an invoice:

    Original names: {product_names}

    Rules:
    1. Correct spelling errors in medicine names
    2. Standardize capitalization (proper case for drug names)
    3. Remove extra characters or OCR artifacts
    4. Keep the same number of items in the same order
    5. If a name seems completely wrong, make your best guess based on context

    Original text context: {raw_text[:500]}...

    Return ONLY a JSON array of corrected names in the same order:
    ["Corrected Name 1", "Corrected Name 2", ...]
    """
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            messages=[{"role": "user", "content": correction_prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        
        response_text = response.choices[0].message.content if response.choices else ""
        corrected_names = json.loads(response_text)
        
        # Apply corrections to products
        for i, product in enumerate(products):
            if i < len(corrected_names):
                if "product" in product:
                    product["product"] = corrected_names[i]
                elif "medicine" in product:
                    product["medicine"] = corrected_names[i]
                else:
                    product["product"] = corrected_names[i]
        
        logging.info(f"Successfully corrected {len(corrected_names)} product names")
        
    except Exception as e:
        logging.error(f"LLM correction failed: {e}")
    
    return products

def create_structured_output(products, invoice_date, other_dates, extracted_text):
    """Create the final structured JSON output with product-date associations"""
    structured_products = []
    
    for product in products:
        product_entry = {
            "product_name": product.get("product") or product.get("medicine") or "Unknown Product"
        }
        
        # Add associated date (priority: expiry > mfg > associated > other dates)
        associated_date = None
        
        if "expiry_date" in product:
            associated_date = product["expiry_date"]
        elif "mfg_date" in product:
            associated_date = product["mfg_date"]
        elif "associated_dates" in product and product["associated_dates"]:
            associated_date = product["associated_dates"][0]
        elif other_dates:
            date_index = len(structured_products) % len(other_dates)
            associated_date = other_dates[date_index]
        
        product_entry["date"] = associated_date
        
        # Add other attributes
        for key, value in product.items():
            if key not in ["product", "medicine", "expiry_date", "mfg_date", "associated_dates"]:
                product_entry[key] = value
        
        structured_products.append(product_entry)
    
    output = {
        "products": structured_products,
        "invoice_date": invoice_date,
        "all_dates_found": other_dates,
        "extracted_text": extracted_text,
        "total_products": len(structured_products)
    }
    
    return output

def contains_json(text):
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False

def extract_first_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            matches = re.findall(r'\{.*?\}', text, re.DOTALL)
            for match in matches:
                try:
                    result = json.loads(match)
                    if isinstance(result, dict) and any(key in result for key in ["medicines", "products", "invoice_date", "other_dates", "extracted_text"]):
                        return result
                except json.JSONDecodeError:
                    continue
                
            potential_json = re.search(r'\{\s*"(?:medicines|products)"\s*:.*?"extracted_text"\s*:.*?\}', text, re.DOTALL)
            if potential_json:
                clean_json = re.sub(r',\s*}', '}', potential_json.group())
                clean_json = re.sub(r',\s*,', ',', clean_json)
                try:
                    return json.loads(clean_json)
                except:
                    pass
        except:
            pass
    
    # Last resort fallback
    try:
        result = {"medicines": [], "invoice_date": None, "other_dates": [], "extracted_text": text}
        
        date_patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',
            r'Date\s*:\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',
            r'Invoice Date\s*:\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
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
    public_id = None
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        file_bytes = np.frombuffer(image_file.read(), np.uint8)

        # Process the image for better OCR results
        processed_image = image_process(file_bytes)
        processed_image.save("output.png")

        # Primary extraction with LayoutLMv3
        layout_result = extract_with_layoutlmv3(processed_image)
        
        if layout_result[0] is not None:  # LayoutLMv3 succeeded
            raw_text, ocr_data, structured_rows, structure_info = layout_result
            logging.info("Using LayoutLMv3 for extraction")
        else:
            # Fallback to EasyOCR
            raw_text, ocr_data, structured_rows = fallback_ocr_extraction(processed_image)
            structure_info = {"layout_detected": False, "fallback_used": True}
            logging.info("Using EasyOCR fallback")
        
        # Extract products with enhanced table analysis
        extracted_products = advanced_table_extraction(structured_rows, structure_info)
        
        # Enhance product names with LLM correction
        corrected_products = enhance_with_llm_correction(extracted_products, raw_text)

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
        
        temperatures = [0.2, 0.5, 0.7, 0.9]

        while retry < maxretry:
            current_temp = temperatures[retry % len(temperatures)]
            
            prompt = (
                """
                You are an expert invoice parser specializing in pharmaceutical and medical invoices.

                Given the text extracted from an invoice image using LayoutLMv3 and pre-processed structured data, extract:

                1. MEDICINES/PRODUCTS: Focus on actual product names, not company names or headers
                2. INVOICE DATE: Main invoice date (format as DD-MM-YYYY)
                3. OTHER DATES: All other dates found (expiry, manufacturing, etc.)

                Pre-extracted structured products from LayoutLMv3 analysis:
                """ + json.dumps(corrected_products, indent=2) + """

                OCR Text with coordinates:
                """ + raw_text + """

                Respond ONLY with valid JSON in this EXACT format:
                {
                  "medicines": [
                    "Product Name 1",
                    "Product Name 2"
                  ],
                  "invoice_date": "DD-MM-YYYY",
                  "other_dates": [
                    "DD-MM-YYYY",
                    "MM-YY"
                  ],
                  "extracted_text": "full extracted text"
                }
                """
                + ("ONLY JSON OUTPUT " * retry)
            )
            
            logging.info(f"Making API call with image URL: {uploaded_image_url}")
            
            try:
                # Verify image URL accessibility
                try:
                    response = requests.head(uploaded_image_url)
                    if response.status_code != 200 or not response.headers.get('Content-Type', '').startswith('image/'):
                        logging.error(f"Image URL is not accessible: {uploaded_image_url}")
                        
                        # Fall back to base64
                        with open("output.png", "rb") as img_file:
                            import base64
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                            img_url = f"data:image/png;base64,{img_data}"
                    else:
                        img_url = uploaded_image_url
                        
                    # API request
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
                    # Fall back to base64
                    with open("output.png", "rb") as img_file:
                        import base64
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        img_url = f"data:image/png;base64,{img_data}"
                    
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

                # Try to extract JSON from response
                extracted_json = extract_first_json(response_text)
                
                if extracted_json:
                    # Always preserve the original OCR text
                    extracted_json['extracted_text'] = raw_text
                    
                    # Ensure all expected fields are present
                    if 'medicines' not in extracted_json:
                        extracted_json['medicines'] = [p.get("product", "Unknown") for p in corrected_products]
                    
                    if 'invoice_date' not in extracted_json:
                        extracted_json['invoice_date'] = None
                    
                    if 'other_dates' not in extracted_json:
                        extracted_json['other_dates'] = []
                    
                    # Calculate quality score
                    score = 0
                    
                    # Score based on medicines found
                    if extracted_json.get('medicines'):
                        score += len(extracted_json['medicines']) * 2
                        valid_medicines = [m for m in extracted_json['medicines'] if m and m.strip() and m.lower() != 'unknown']
                        score += len(valid_medicines) * 3
                    
                    # Score for invoice date
                    if extracted_json.get('invoice_date'):
                        score += 5
                    
                    # Score for other dates
                    if extracted_json.get('other_dates'):
                        score += len(extracted_json['other_dates']) * 2
                    
                    # Score for extracted text completeness
                    if extracted_json.get('extracted_text') and len(extracted_json['extracted_text']) > 100:
                        score += 3
                    
                    logging.info(f"Response quality score: {score}")
                    
                    if score > best_score:
                        best_score = score
                        best_response = response_text
                        best_json = extracted_json
                        
                        # If we have a good enough response, use it
                        if score >= 15:  # Threshold for acceptable response
                            break
                
            except Exception as api_error:
                logging.error(f"API call failed on retry {retry}: {api_error}")
            
            retry += 1

        # Use the best response found
        if best_json:
            final_result = best_json
        else:
            # Final fallback using the structured products we already extracted
            final_result = create_structured_output(
                corrected_products, 
                None,  # invoice_date
                [],    # other_dates
                raw_text
            )

        # Clean up uploaded image
        if public_id:
            cleanup_cloudinary_file(public_id)

        # Clean up local files
        try:
            os.remove("output.png")
        except:
            pass

        return jsonify(final_result)

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        
        # Clean up on error
        if public_id:
            cleanup_cloudinary_file(public_id)
        
        try:
            os.remove("output.png")
        except:
            pass
        
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "layoutlmv3_available": layoutlm_processor is not None,
        "easyocr_available": easyocr_reader is not None,
        "together_api_configured": TOGETHER_API_KEY is not None,
        "cloudinary_configured": all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET])
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
