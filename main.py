from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
import os
import fitz
import numpy as np
import cv2
import s3_utils
import logging
import json
import io

# --- Immediate debug print ---
# print("Starting execution in", os.path.abspath(__file__))

# --- Test logging at module level ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Logger initialized successfully in %s", os.path.abspath(__file__))

# --- Load .env ---
load_dotenv()
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_BUCKET_ACCESS_KEY = os.getenv("S3_BUCKET_ACCESS_KEY")
S3_BUCKET_SECRET_KEY = os.getenv("S3_BUCKET_SECRET_KEY")

# Validate S3 credentials
if not all([S3_BUCKET_NAME, S3_BUCKET_ACCESS_KEY, S3_BUCKET_SECRET_KEY]):
    raise ValueError("Missing S3 credentials in .env")

# --- Database setup ---
DATABASE_URL = f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Updated Model with single JSON column ---
class PDFCropImages(Base):
    __tablename__ = "pdf_crop_images1"
    
    id = Column(Integer, primary_key=True, index=True)
    class_id = Column(Integer, nullable=False)
    subject_id = Column(Integer, nullable=False)
    course_id = Column(Integer, nullable=False)
    module_id = Column(Integer, nullable=False)
    
    # Single JSON column to store all categories and their URLs
    # Structure: {
    #   "tables": ["url1", "url2"],
    #   "equations": ["url3", "url4"],
    #   "diagrams": ["url5"],
    #   "others": []
    # }
    image_urls = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create table if not exists
Base.metadata.create_all(bind=engine)

# --- FastAPI app ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

UPLOAD_DIR = "uploaded-pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Utility functions ---
def get_default_image_urls_structure():
    """Returns the default structure for image_urls JSON"""
    return {
        "tables": [],
        "equations": [],
        "diagrams": [],
        "others": []
    }

def render_page(pdf_path, page_num, zoom=2):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return np.array(img)

# --- API Endpoints ---

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=422, detail="Only PDF files are allowed")
    filename = file.filename
    save_path = os.path.join(UPLOAD_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    return {"status": "uploaded", "filename": filename}

@app.get("/get-page")
def get_page(page: int, pdf_name: str):
    pdf_path = os.path.join(UPLOAD_DIR, pdf_name)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    img = render_page(pdf_path, page)
    _, im_buf_arr = cv2.imencode(".png", img)
    return StreamingResponse(io.BytesIO(im_buf_arr.tobytes()), media_type="image/png")

@app.get("/get-total-pages")
def get_total_pages(pdf_name: str):
    pdf_path = os.path.join(UPLOAD_DIR, pdf_name)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()
    return {"total_pages": total_pages}

@app.post("/save-crop/")
async def save_crop(
    file: UploadFile,
    page: int = Form(...),
    category: str = Form(...),
    pdf_name: str = Form(...),
    class_id: int = Form(...),
    subject_id: int = Form(...),
    course_id: int = Form(...),
    module_id: int = Form(...),
    folder: str = Form(...)
):
    # print("Entered save_crop endpoint")
    try:
        logger.info(f"Received request with: category={category}, page={page}, pdf_name={pdf_name}, "
                    f"class_id={class_id}, subject_id={subject_id}, course_id={course_id}, module_id={module_id}, folder={folder}")

        valid_categories = ["equations", "diagrams", "tables", "others"]
        if category not in valid_categories:
            raise HTTPException(status_code=422, detail=f"Invalid category. Must be one of {valid_categories}")

        if not file.size:
            raise HTTPException(status_code=422, detail="Uploaded file is empty")

        if not pdf_name.endswith('.pdf'):
            raise HTTPException(status_code=422, detail="Invalid PDF name")

        image_bytes = await file.read()
        # print(f"Image bytes length: {len(image_bytes)}")
        
        # Upload to S3
        try:
            s3_response = s3_utils.upload_to_s3(
                file_data=image_bytes,
                original_filename=f"{folder}/crop_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                bucket_name=S3_BUCKET_NAME
            )
            s3_url = s3_response["Location"]
            # print(f"S3 upload successful, URL: {s3_url}")
        except Exception as s3_error:
            # print(f"S3 upload failed: {str(s3_error)}")
            raise HTTPException(status_code=500, detail=f"S3 upload error: {str(s3_error)}")

        db = SessionLocal()
        try:
            # Use a more explicit approach with row locking
            existing_record = db.query(PDFCropImages).filter(
                PDFCropImages.class_id == class_id,
                PDFCropImages.subject_id == subject_id,
                PDFCropImages.course_id == course_id,
                PDFCropImages.module_id == module_id
            ).with_for_update().first()  # Add row-level locking

            if existing_record:
                logger.info(f"Updating existing record ID: {existing_record.id}")
                
                # Fetch fresh data from database to avoid stale data
                db.refresh(existing_record)
                
                # Get current image_urls or initialize with default structure
                current_image_urls = existing_record.image_urls or get_default_image_urls_structure()
                
                # Deep copy to ensure we're working with a new object
                import copy
                updated_image_urls = copy.deepcopy(current_image_urls)
                
                # Ensure the category exists in the structure
                if category not in updated_image_urls:
                    updated_image_urls[category] = []
                
                # Add new URL to the category
                updated_image_urls[category].append(s3_url)
                
                print(f"Before update - Current URLs: {current_image_urls}")
                print(f"After update - New URLs: {updated_image_urls}")
                
                # Update the record with the new structure
                existing_record.image_urls = updated_image_urls
                existing_record.updated_at = datetime.utcnow()
                
                # Force the ORM to recognize the change
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(existing_record, "image_urls")
                
            else:
                logger.info(f"Creating new record for class_id={class_id}, subject_id={subject_id}, course_id={course_id}, module_id={module_id}")
                
                # Initialize the image_urls structure
                image_urls_structure = get_default_image_urls_structure()
                image_urls_structure[category].append(s3_url)
                
                new_entry = PDFCropImages(
                    class_id=class_id,
                    subject_id=subject_id,
                    course_id=course_id,
                    module_id=module_id,
                    image_urls=image_urls_structure
                )
                db.add(new_entry)
                # print(f"Created new record with structure: {image_urls_structure}")

            # Explicit flush before commit
            db.flush()
            db.commit()
            
            # Verify the update was successful
            verification_record = db.query(PDFCropImages).filter(
                PDFCropImages.class_id == class_id,
                PDFCropImages.subject_id == subject_id,
                PDFCropImages.course_id == course_id,
                PDFCropImages.module_id == module_id
            ).first()
            
            if verification_record:
                # print(f"Verification - Final image_urls: {verification_record.image_urls}")
                logger.info(f"Successfully saved image with URL: {s3_url}")
            else:
                logger.error("Verification failed - record not found after commit")
            
        except Exception as db_error:
            db.rollback()
            # print(f"Database error occurred: {str(db_error)}")
            logger.error(f"Database commit failed: {str(db_error)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
        finally:
            db.close()

        return {
            "status": "success", 
            "s3_url": s3_url, 
            "message": f"✅ Image URL added to {category} category in DB."
        }

    except HTTPException as e:
        logger.error(f"HTTPException: {str(e.detail)}")
        raise e
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/get-images/")
def get_images(class_id: int, subject_id: int, course_id: int, module_id: int):
    db = SessionLocal()
    try:
        record = db.query(PDFCropImages).filter(
            PDFCropImages.class_id == class_id,
            PDFCropImages.subject_id == subject_id,
            PDFCropImages.course_id == course_id,
            PDFCropImages.module_id == module_id
        ).first()
        
        if not record:
            return get_default_image_urls_structure()
        
        # Return the image_urls JSON structure
        image_urls = record.image_urls or get_default_image_urls_structure()
        
        return {
            "image_urls": image_urls,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "total_images": sum(len(urls) for urls in image_urls.values())
        }
        
    except Exception as e:
        logger.error(f"Error retrieving images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving images: {str(e)}")
    finally:
        db.close()

# --- Endpoint to get images by specific category ---
@app.get("/get-images-by-category/")
def get_images_by_category(
    class_id: int, 
    subject_id: int, 
    course_id: int, 
    module_id: int, 
    category: str
):
    valid_categories = ["equations", "diagrams", "tables", "others"]
    if category not in valid_categories:
        raise HTTPException(status_code=422, detail=f"Invalid category. Must be one of {valid_categories}")
    
    db = SessionLocal()
    try:
        record = db.query(PDFCropImages).filter(
            PDFCropImages.class_id == class_id,
            PDFCropImages.subject_id == subject_id,
            PDFCropImages.course_id == course_id,
            PDFCropImages.module_id == module_id
        ).first()
        
        if not record or not record.image_urls:
            return {category: []}
        
        return {
            category: record.image_urls.get(category, []),
            "count": len(record.image_urls.get(category, []))
        }
        
    except Exception as e:
        logger.error(f"Error retrieving images by category: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving images: {str(e)}")
    finally:
        db.close()