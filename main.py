from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional
from PIL import Image
import os
import io
import fitz
import numpy as np
import cv2

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded-pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Utility: Render PDF page as image
def render_page(pdf_path, page_num, zoom=2):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(img)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile):
    filename = file.filename
    save_path = os.path.join(UPLOAD_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    return {"status": "uploaded", "filename": filename}

@app.get("/get-page")
def get_page(page: int, pdf_name: str):
    pdf_path = os.path.join(UPLOAD_DIR, pdf_name)
    if not os.path.exists(pdf_path):
        return JSONResponse(status_code=404, content={"error": "PDF not found"})
    img = render_page(pdf_path, page)
    _, im_buf_arr = cv2.imencode(".png", img)
    return StreamingResponse(io.BytesIO(im_buf_arr.tobytes()), media_type="image/png")

# @app.post("/save-crop/")
# async def save_crop(
#     file: UploadFile,
#     page: int = Form(...),
#     folder: str = Form(...),
#     pdf_name: str = Form(...),
#     output_dir: str = "images-lech204"
# ):
#     folder_path = os.path.join(output_dir, folder)
#     os.makedirs(folder_path, exist_ok=True)

#     image_data = await file.read()
#     img = Image.open(io.BytesIO(image_data))

#     filename = f"{os.path.splitext(pdf_name)[0]}_page_{page+1}_{folder}_{len(os.listdir(folder_path))+1}.png"
#     img.save(os.path.join(folder_path, filename))

#     return {"status": "success", "saved_as": filename}


@app.post("/save-crop/")
async def save_crop(
    file: UploadFile,
    page: int = Form(...),
    folder: str = Form(...),
    pdf_name: str = Form(...),
):
    output_dir = f"images--{os.path.splitext(pdf_name)[0]}"  # Dynamic folder name

    folder_path = os.path.join(output_dir, folder)
    os.makedirs(folder_path, exist_ok=True)

    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data))

    filename = f"{os.path.splitext(pdf_name)[0]}_page_{page+1}_{folder}_{len(os.listdir(folder_path)) + 1}.png"
    img.save(os.path.join(folder_path, filename))

    return {"status": "success", "saved_as": filename}


@app.get("/get-total-pages")
def get_total_pages(pdf_name: str):
    pdf_path = os.path.join(UPLOAD_DIR, pdf_name)
    if not os.path.exists(pdf_path):
        return JSONResponse(status_code=404, content={"error": "PDF not found"})
    try:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()
        return {"total_pages": total_pages}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})