import os
import io
import re
import json
import fitz  # PyMuPDF
import numpy as np
import easyocr
import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

# Carrega variáveis de ambiente
load_dotenv()

app = FastAPI()

# Inicializa OCR (Português e Inglês)
reader = easyocr.Reader(['pt', 'en'], gpu=False)

class ItemData(BaseModel):
    descricao: str
    preco_unitario: float
    quantidade: float
    taxa_iva_percentagem: float

class DocumentData(BaseModel):
    supplier_name: str
    nif: Optional[str]
    invoice_number: str
    data_emissao: str
    valor_total_documento: float
    total_iva: float
    valor_pago: float
    items: List[ItemData]

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

parser = JsonOutputParser(pydantic_object=DocumentData)

def limpar_texto(texto: str) -> str:
    texto = re.sub(r'[^\x00-\x7F]+', ' ', texto)
    texto = re.sub(r'\s{2,}', ' ', texto)
    return texto.strip()

def corrigir_numeros(texto: str) -> str:
    """
    Corrige problemas comuns do OCR:
    - Remove pontos de milhar
    - Converte vírgula decimal para ponto
    """
    # Remove pontos usados como separador de milhar: 3.245 -> 3245
    texto = re.sub(r'(\d)\.(\d{3})(?!\d)', r'\1\2', texto)
    # Troca vírgula por ponto nos decimais: 3245,61 -> 3245.61
    texto = re.sub(r'(\d+),(\d{2})', r'\1.\2', texto)
    return texto

def validar_e_corrigir_dados(data: dict, texto_ocr: str) -> dict:
    try:
        # Detecta taxa IVA geral no texto OCR se todos os itens vierem com 0
        if data.get("items"):
            taxas = {item["taxa_iva_percentagem"] for item in data["items"]}
            if taxas == {0}:  # Todos os itens têm IVA 0
                match = re.search(r"TAXA%[^\d]*(\d{1,2})", texto_ocr, re.IGNORECASE)
                if match:
                    taxa_detectada = float(match.group(1))
                    for item in data["items"]:
                        item["taxa_iva_percentagem"] = taxa_detectada

            # Recalcula subtotais
            subtotal = round(sum(item["preco_unitario"] * item["quantidade"] for item in data["items"]), 2)
            iva_calc = round(sum(
                (item["preco_unitario"] * item["quantidade"]) * (item["taxa_iva_percentagem"] / 100)
                for item in data["items"]
            ), 2)
            total_calc = round(subtotal + iva_calc, 2)

            # Atualiza campos principais
            data["total_iva"] = iva_calc
            data["valor_total_documento"] = total_calc
    except Exception as e:
        print(f"[VALIDAÇÃO] Erro ao validar: {e}")
    return data

def preprocess_image(img):
    """Melhora contraste e remove ruído antes do OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

@app.post("/ocr")
async def perform_ocr_and_data_extraction(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Formato de arquivo não suportado.")

    extracted_text = ""

    try:
        file_content = await file.read()

        if file.filename.lower().endswith('.pdf'):
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap(dpi=300)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                img_upscaled = cv2.resize(img_array, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                img_preprocessed = preprocess_image(img_upscaled)
                results = reader.readtext(img_preprocessed)
                filtered = [text for (_, text, conf) in results if conf > 0.4]
                extracted_text += " ".join(filtered) + " "
        else:
            img_bytes = np.frombuffer(file_content, np.uint8)
            img_array = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            img_upscaled = cv2.resize(img_array, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            img_preprocessed = preprocess_image(img_upscaled)
            results = reader.readtext(img_preprocessed)
            filtered = [text for (_, text, conf) in results if conf > 0.4]
            extracted_text = " ".join(filtered)


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento do arquivo: {str(e)}")

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Nenhum texto extraído do documento.")

    try:
        prompt = f"""
Você é um assistente de OCR. Extraia todos os dados do documento abaixo e **retorne somente o JSON válido** (sem explicações ou comentários).
O texto foi extraído por OCR e pode conter erros. Sua tarefa é inferir com absoluta precisão os valores corretos.

Responda apenas com o JSON no formato abaixo (sem texto extra):
{parser.get_format_instructions()}

Instruções importantes:
- NUNCA invente dados.
- Se um campo não estiver claro, retorne null ou 0.
- NÃO reutilize exemplos de instruções anteriores.
- Identifique valores reais no texto e siga a estrutura JSON com atenção.
- Os valores numéricos devem corresponder exatamente ao que está escrito, sem arredondamentos indevidos.
- Sempre use dois decimais para valores monetários.
- Caso veja valores como "3 245,61" ou "3245,61", converta para 3245.61 mantendo todos os dígitos.
- O campo "valor_total_documento" deve representar o valor total COM IVA.
- O campo "total_iva" deve conter a soma total dos valores de IVA de todos os itens.
- Caso todos os produtos tenham a mesma taxa de IVA, aplique-a a todos os itens.
- Liste todos os produtos/serviços exatamente como aparecem no documento, mesmo que tenham nomes semelhantes.
- NUNCA agrupe ou resuma produtos.
- Use ponto como separador decimal.
- A lista de itens pode conter vários produtos. Certifique-se de extrair TODOS os produtos da lista, seguindo o mesmo formato de objeto para cada um.

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

        # Log para debug
        print("\n===== Texto OCR Extraído =====")
        print(extracted_text[:1000], "...")
        print("\n===== Resposta bruta do LLM =====")
        print(response.content)

        try:
            extracted_data = parser.parse(response.content)
        except Exception:
            # Fallback: tenta limpar e parsear manualmente
            cleaned = re.search(r"\{.*\}", response.content, re.S)
            if not cleaned:
                raise HTTPException(status_code=500, detail="Não foi possível encontrar JSON na resposta.")
            try:
                extracted_data = json.loads(cleaned.group())
            except json.JSONDecodeError as je:
                raise HTTPException(status_code=500, detail=f"Falha ao decodificar JSON: {str(je)}")

        extracted_data = validar_e_corrigir_dados(extracted_data, extracted_text)
        return {
            "extracted_text": extracted_text,
            "extracted_data": extracted_data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro de extração com LLM: {str(e)}")
