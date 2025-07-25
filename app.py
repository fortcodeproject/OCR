from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageEnhance, UnidentifiedImageError
from io import BytesIO
import fitz  # PyMuPDF
import easyocr

app = FastAPI()

# Inicializa o leitor do EasyOCR uma vez
reader = easyocr.Reader(['pt'], gpu=False)

def preprocess_image(image: Image.Image) -> Image.Image:
    """Melhora o contraste para OCR."""
    image = image.convert("L")  # escala de cinza
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(2.0)

def ocr_image(image: Image.Image) -> str:
    """Executa OCR com EasyOCR."""
    image = preprocess_image(image)
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    result = reader.readtext(img_bytes.read(), detail=0)
    return "\n".join(result).strip()

def extract_text(file_bytes: bytes, filename: str, content_type: str) -> str:
    try:
        if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
            text = ""
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num, page in enumerate(doc, 1):
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = ocr_image(img)
                text += f"\n--- Página {page_num} ---\n{page_text}"
            doc.close()
            return text.strip()
        elif content_type.startswith("image/"):
            img = Image.open(BytesIO(file_bytes))
            return ocr_image(img)
        else:
            return "Tipo de arquivo não suportado para OCR."
    except UnidentifiedImageError:
        return "Erro: Não foi possível identificar a imagem."
    except Exception as e:
        return f"Erro durante OCR: {str(e)}"

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    file_bytes = await file.read()
    text = extract_text(file_bytes, file.filename, file.content_type)
    return {"extracted_text": text}
