import os
import io
import re
import fitz # PyMuPDF
import numpy as np
import easyocr
import cv2 # Importe o OpenCV
from dotenv import load_dotenv
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

# Carrega as variáveis de ambiente (.env)
load_dotenv()

# Inicializa o FastAPI
app = FastAPI()

# Inicializa o EasyOCR (idiomas português e inglês, sem GPU)
reader = easyocr.Reader(['pt', 'en'], gpu=False)

# Define a estrutura dos dados extraídos do documento
class ItemData(BaseModel):
    descricao: str = Field(description="A descrição do produto ou serviço.")
    preco_unitario: float = Field(description="O preço unitário do item.")
    quantidade: float = Field(description="A quantidade do item.")
    taxa_iva_percentagem: float = Field(description="A taxa de IVA aplicada ao item.")

class DocumentData(BaseModel):
    supplier_name: str = Field(description="Nome da empresa ou fornecedor.")
    nif: Optional[str] = Field(description="NIF da empresa, se disponível.")
    invoice_number: str = Field(description="Número da fatura.")
    data_emissao: str = Field(description="Data de emissão (dd-mm-yyyy).")
    valor_total_documento: float = Field(description="Valor total do documento.")
    total_iva: float = Field(description="Valor total do IVA.")
    valor_pago: float = Field(description="Valor pago.")
    items: List[ItemData] = Field(description="Lista de produtos/serviços.")

# Inicializa o modelo LLM Groq
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Parser JSON com base no Pydantic
parser = JsonOutputParser(pydantic_object=DocumentData)

# Função para limpar texto OCR ruidoso
def limpar_texto(texto: str) -> str:
    texto = re.sub(r'[^\x00-\x7F]+', ' ', texto)  # remove caracteres não ASCII
    texto = re.sub(r'\s{2,}', ' ', texto)          # remove espaços múltiplos
    return texto.strip()

@app.post("/ocr")
async def perform_ocr_and_data_extraction(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Formato de arquivo não suportado.")

    extracted_text = ""

    try:
        file_content = await file.read()

        if file.filename.endswith('.pdf'):
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap(dpi=300) # Renderiza o PDF com alta DPI
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                
                # Opcional: Aumentar a resolução da imagem do PDF
                img_upscaled = cv2.resize(img_array, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                
                results = reader.readtext(img_upscaled)
                filtered = [text for (_, text, conf) in results if conf > 0.3]
                extracted_text += " ".join(filtered) + " "
        else:
            img_bytes = np.frombuffer(file_content, np.uint8)
            img_array = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

            # Aumentar a resolução da imagem
            img_upscaled = cv2.resize(img_array, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            results = reader.readtext(img_upscaled)
            filtered = [text for (_, text, conf) in results if conf > 0.3]
            extracted_text = " ".join(filtered)

        extracted_text = limpar_texto(extracted_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento do arquivo: {str(e)}")

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Nenhum texto extraído do documento.")

    try:
        prompt = f"""
Você é um assistente de OCR. Extraia os dados do documento abaixo e **retorne somente o JSON válido** (sem explicações ou comentários).
O texto foi extraído por OCR e pode conter erros. Sua tarefa é inferir os valores corretos.

Responda apenas com o JSON no formato abaixo (sem texto extra):
{parser.get_format_instructions()}

⚠️ Instruções:
- NUNCA invente dados.
- Se um campo não estiver claro, retorne null ou 0.
- NÃO reutilize exemplos de instruções anteriores.
- Identifique valores reais no texto e siga a estrutura JSON com atenção.

Retorne estritamente neste formato:

{{
  "supplier_name": "...",
  "nif": "... ou null",
  "invoice_number": "...",
  "data_emissao": "dd-mm-yyyy",
  "valor_total_documento": ...,
  "total_iva": ...,
  "valor_pago": ...,
  "items": [
    {{
      "descricao": "...",
      "preco_unitario": ...,
      "quantidade": ...,
      "taxa_iva_percentagem": ...
    }}
  ]
}}

Texto extraído do documento:
{extracted_text}
        """

        response = llm.invoke(prompt)
        extracted_data = parser.parse(response.content)

        return {
            "extracted_text": extracted_text,
            "extracted_data": extracted_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro de extração com LLM: {str(e)}")

# Para rodar a API, use: uvicorn app:app --reload