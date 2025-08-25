import os
import re
import io
import json
import unicodedata
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query
from pydantic import BaseModel
from typing import Optional, List, Tuple, Dict, Any

# LangChain + Groq
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

# =========================
# Config / Inicialização
# =========================
load_dotenv()

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

OCR_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_THREADS = int(os.getenv("OCR_THREADS", str(max(2, (os.cpu_count() or 4) // 2))))
OCR_LANGS = os.getenv("OCR_LANGS", "por+eng")
OCR_PSM = os.getenv("OCR_PSM", "6")
LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama3-70b-8192")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
MAX_CHARS_TO_LLM = int(os.getenv("MAX_CHARS_TO_LLM", "120000"))

TESSERACT_CONFIG = rf'--oem 3 --psm {OCR_PSM} -l {OCR_LANGS}'

app = FastAPI(title="OCR + Extração Estruturada Turbo", version="1.1.0")

# =========================
# Modelos
# =========================
class ItemData(BaseModel):
    descricao: str
    preco_unitario: float
    quantidade: float
    taxa_iva_percentagem: float

class DocumentData(BaseModel):
    supplier_name: str
    nif: Optional[str] = None
    invoice_number: str
    data_emissao: str
    valor_total_documento: float
    total_iva: float
    valor_pago: float
    items: List[ItemData]

parser = JsonOutputParser(pydantic_object=DocumentData)

llm = ChatGroq(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    api_key=os.getenv("GROQ_API_KEY"),
)

# =========================
# Helpers de texto/número
# =========================
def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def canon(s: str) -> str:
    return strip_accents(s).upper()

def limpar_texto(texto: str) -> str:
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

def limpar_e_ajustar_texto_para_llm(texto_ocr: str) -> str:
    """
    Limpa o texto do OCR e faz ajustes heurísticos antes de enviar para o LLM.
    """
    # 1. Corrigir erros comuns de OCR (ex: 2006 -> 200G)
    # A string 'CHOURICO 2006' deve ser substituída por 'CHOURICO 200G'
    # Use re.sub para ser insensível a maiúsculas/minúsculas e espaços
    texto_limpo = re.sub(r'chourico\s+2006', 'CHOURICO 200G', texto_ocr, flags=re.IGNORECASE)

    # 2. Heurística para cortar a secção do rodapé confusa
    # Procurar por uma linha-chave (como "DETALHES DO CLIENTE") e cortar o texto depois dela,
    # uma vez que o LLM já tem os itens e o rodapé é onde estão os números errados da quantidade.
    # Esta é uma estratégia avançada mas eficaz para evitar a confusão do LLM.
    cut_off_pattern = r'DETALHES\s+DO\s+CLIENTE'
    match = re.search(cut_off_pattern, texto_limpo, re.IGNORECASE)
    if match:
        texto_limpo = texto_limpo[:match.start()]

    return limpar_texto(texto_limpo)

def to_float(num_str: str) -> float:
    """
    Converte strings como '106.946,40' | '106 946,40' | '106,946.40' | '5092,69' em float.
    """
    if not num_str:
        return 0.0
    s = num_str.strip().replace('\u00A0', ' ')
    s = re.sub(r'[^\d,.\-]', '', s)  # mantém apenas dígitos, vírgula, ponto e sinal
    if s.count(',') and s.count('.'):
        # Assume a ÚLTIMA ocorrência como separador decimal
        last_comma = s.rfind(',')
        last_dot = s.rfind('.')
        if last_comma > last_dot:
            s = s.replace('.', '')      # ponto = milhar
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')      # vírgula = milhar
    else:
        # Só vírgula => decimal
        if s.count(','):
            s = s.replace(',', '.')
        # Só ponto => já decimal
    try:
        return float(s)
    except ValueError:
        return 0.0

def find_number_after_label(text: str, labels: List[str]) -> Optional[float]:
    """
    Procura um número logo após algum dos rótulos (robusto a quebras e espaços).
    """
    t = canon(text)
    for label in labels:
        pattern = rf'{label}\s*[:\-]?\s*([0-9\.\,\s]+)'
        m = re.search(pattern, t, flags=re.IGNORECASE)
        if m:
            return to_float(m.group(1))
    return None

# =========================
# OCR / PDF
# =========================
def is_meaningful(texto: str) -> bool:
    if not texto:
        return False
    t = texto.strip()
    return len(t) > 40 and bool(re.search(r'\d', t))

def pixmap_to_numpy(pix: fitz.Pixmap) -> np.ndarray:
    if pix.n not in (3, 4):
        pix = fitz.Pixmap(fitz.csRGB, pix)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    if pix.n == 4:
        img = img.reshape(pix.height, pix.width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = img.reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def ocr_image(idx_img: Tuple[int, np.ndarray]) -> Tuple[int, str]:
    idx, img = idx_img
    text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
    return idx, text

def extract_text_from_pdf_stream(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    n = doc.page_count
    candidates_embedded: List[Tuple[int, str]] = []
    candidates_ocr_imgs: List[Tuple[int, np.ndarray]] = []

    for i in range(n):
        page = doc.load_page(i)
        txt = page.get_text("text")
        if is_meaningful(txt):
            candidates_embedded.append((i, txt))
            continue
        pix = page.get_pixmap(dpi=OCR_DPI)
        img_bgr = pixmap_to_numpy(pix)
        img_bin = preprocess_image(img_bgr)
        candidates_ocr_imgs.append((i, img_bin))

    ocr_results: List[Tuple[int, str]] = []
    if candidates_ocr_imgs:
        with ThreadPoolExecutor(max_workers=OCR_THREADS) as ex:
            futures = [ex.submit(ocr_image, tup) for tup in candidates_ocr_imgs]
            for fut in as_completed(futures):
                ocr_results.append(fut.result())

    all_results = {i: limpar_texto(t) for i in candidates_embedded}
    for i, t in ocr_results:
        all_results[i] = limpar_texto(t)

    ordered_text = "\n".join(all_results[i] for i in sorted(all_results.keys()))
    return ordered_text

def extract_text_from_image_stream(file_bytes: bytes) -> str:
    arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return ""
    img_bin = preprocess_image(img_bgr)
    text = pytesseract.image_to_string(img_bin, config=TESSERACT_CONFIG)
    return limpar_texto(text)

# =========================
# Heurísticas de totais/IVA
# =========================
def extract_totals_from_text(extracted_text: str) -> Dict[str, float]:
    """
    Lê rodapés comuns de faturas:
      - TOTAL (KZ) / TOTAL GERAL / TOTAL A PAGAR => total com IVA
      - TOTAL IMPOSTOS / TOTAL IVA / IVA => total_iva
      - TOTAL LÍQUIDO => base sem IVA
      - INCIDÊNCIA ... TAXA% ... VALOR (tabela inferida)
    """
    out = {"total_com_iva": None, "total_iva": None, "total_liquido": None, "taxa_padrao": None}
    t = canon(extracted_text)

    # TOTAL COM IVA (preferência por rótulos inequívocos)
    labels_total = [
        r'TOTAL\s*\(KZ\)', r'TOTAL\s+GERAL', r'TOTAL\s+A\s+PAGAR', r'TOTAL\s+A\s+LIQUIDAR',
        r'TOTAL\s+PAGO', r'VALOR\s+A\s+PAGAR', r'VALOR\s+PAGO'
    ]
    for lab in labels_total:
        m = re.search(rf'{lab}\s*[:\-]?\s*([0-9\.\,\s]+)', t, flags=re.IGNORECASE)
        if m:
            out["total_com_iva"] = to_float(m.group(1))
            break

    # TOTAL IMPOSTOS / IVA
    labels_iva = [
        r'TOTAL\s+IMPOSTOS', r'TOTAL\s+IVA', r'IVA\b', r'IMPOSTOS\b', r'IMPOSTO\b'
    ]
    for lab in labels_iva:
        m = re.search(rf'{lab}\s*[:\-]?\s*([0-9\.\,\s]+)', t, flags=re.IGNORECASE)
        if m:
            out["total_iva"] = to_float(m.group(1))
            break

    # TOTAL LÍQUIDO (base)
    m = re.search(r'TOTAL\s+LIQUIDO\s*[:\-]?\s*([0-9\.\,\s]+)', t, flags=re.IGNORECASE)
    if m:
        out["total_liquido"] = to_float(m.group(1))

    # Padrão "INCIDENCIA ... TAXA% ... VALOR"
    m = re.search(
        r'INCID[ÊE]NCIA.*?([0-9\.\,\s]+).*?TAXA\s*%?\s*[:\-]?\s*([0-9]{1,2})(?:[,\.\s]\d+)?\s*%?.*?(?:VALOR|IMPOSTOS).*?([0-9\.\,\s]+)',
        t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        base = to_float(m.group(1))
        taxa = float(m.group(2))
        val_iva = to_float(m.group(3))
        # confere coerência básica
        if base > 0 and val_iva > 0:
            out["taxa_padrao"] = taxa
            out["total_liquido"] = out["total_liquido"] or base
            out["total_iva"] = out["total_iva"] or val_iva
            out["total_com_iva"] = out["total_com_iva"] or round(base + val_iva, 2)

    # Se não achou "TOTAL (KZ)" mas tem liquido + impostos, soma
    if out["total_com_iva"] is None and (out["total_liquido"] is not None and out["total_iva"] is not None):
        out["total_com_iva"] = round(out["total_liquido"] + out["total_iva"], 2)

    return out

# =========================
# LLM
# =========================
def build_prompt_for_llm(extracted_text: str) -> str:
    return f"""
Você é um assistente de OCR. Extraia os dados do documento abaixo e **retorne SOMENTE o JSON válido** (sem explicações).
O texto pode conter erros de OCR; infira com precisão, mas não invente.

Instruções Cruciais para Itens da Fatura:
- Para cada item, extraia a 'descricao', o 'preco_unitario' (preço por unidade, SEM IVA), a 'quantidade' e a 'taxa_iva_percentagem'.
- Se a fatura apresentar um 'valor total da linha' para o item (preço * quantidade) mas não o 'preco_unitario', **divida o 'valor total da linha' pela 'quantidade'** para obter o 'preco_unitario'.
- Certifique-se de que cada item seja um objeto distinto dentro da lista 'items'.
- ATENÇÃO ÀS TABELAS: Para extrair a 'quantidade', **identifique a coluna 'Quantidade'** na tabela. Ignore números de outras seções do documento (como taxas de IVA, NIFs, etc.) que não estejam na coluna de quantidade dos itens.
- NÃO arredonde os valores de 'preco_unitario' ou 'quantidade' no JSON. Apenas formate como float.

Formato Pydantic:
{parser.get_format_instructions()}

Atenção a rótulos equivalentes:
- TOTAL (KZ), TOTAL GERAL, TOTAL A PAGAR, TOTAL A LIQUIDAR ⇒ "valor_total_documento" e, se houver, "valor_pago"
- TOTAL IMPOSTOS, TOTAL IVA, IVA ⇒ "total_iva"
- TOTAL LÍQUIDO ⇒ base sem IVA
- TAXA% ou TAXA (sem %) ⇒ "taxa_iva_percentagem" dos itens (se for uma taxa única)
- Use ponto como separador decimal.
- Datas dd-mm-yyyy.
- Não agrupe itens

Restrições:
- Nunca invente dados que não estejam presentes no documento.
- Retorne estritamente o JSON.

TEXTO:
{extracted_text}
""".strip()

def run_llm_structured_extraction(extracted_text: str) -> Dict[str, Any]:
    processed_text = limpar_e_ajustar_texto_para_llm(extracted_text)
    text_for_llm = processed_text if len(processed_text) <= MAX_CHARS_TO_LLM else processed_text[:MAX_CHARS_TO_LLM]

    prompt = build_prompt_for_llm(text_for_llm)
    response = llm.invoke(prompt)

    # Debug opcional
    print("\n===== AMOSTRA TEXTO OCR =====")
    print(text_for_llm[:1000], "...\n")
    print("===== RESPOSTA LLM =====")
    print(response.content[:1000], "...\n")

    try:
        extracted_data = parser.parse(response.content)
    except Exception:
        m = re.search(r"\{.*\}", response.content, re.S)
        if not m:
            raise HTTPException(status_code=500, detail="LLM não retornou JSON válido.")
        extracted_data = json.loads(m.group())

    return extracted_data

# =========================
# Pós-processamento (fixes)
# =========================
# helpers adicionais
def safe_float(x: Any, default: float = 0.0) -> float:
    """
    Converte vários tipos (str com vírgula/ponto, int, float, None) para float.
    Usa to_float() para strings não-numéricas.
    """
    try:
        if x is None:
            return float(default)
        if isinstance(x, float):
            return x
        if isinstance(x, int):
            return float(x)
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return float(default)
            # se for já algo como '1234.56' ou '1234,56'
            return to_float(s)
        # fallback
        return float(x)
    except Exception:
        return float(default)


def first_present(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def validar_e_corrigir_dados(data: Dict[str, Any], texto_ocr: str) -> Dict[str, Any]:
    """
    Pós-processamento robusto e heurístico para:
      - Normalizar nomes de campos de itens (preco_unitario, preco_total, quantidade, taxa).
      - Detectar quando o LLM retornou preço total por item em vez de unitário e converter.
      - Aplicar taxa padrão detectada no rodapé quando itens tiverem taxa 0/ausente.
      - Recalcular subtotal, IVA e total, priorizando valores do rodapé quando disponíveis.
      - Garantir arredondamento e formatação corretos.
    """
    try:
        if not data:
            return data

        # Detecção do rodapé (taxa padrão / totais)
        det = extract_totals_from_text(texto_ocr)
        taxa_padrao = det.get("taxa_padrao")

        items = data.get("items") or []
        normalized_items = []

        # Normalizar itens: aceitar vários nomes que o LLM pode ter usado
        for it in items:
            # it pode já ser dict ou pydantic. Vamos transformá-lo num dict mutável
            if not isinstance(it, dict):
                try:
                    it = it.dict()
                except Exception:
                    it = dict(it)

            # Capturar possíveis chaves alternativas
            descricao = first_present(it, ["descricao", "descrição", "descricao_item", "desc", "descricao_produto"], "")
            # preços: preco_unitario esperado; mas LLM pode devolver "preco_total", "valor", "subtotal"
            preco_unit = first_present(it, ["preco_unitario", "preco_unit", "unit_price", "preco_unitario_item"], None)
            preco_total = first_present(it, ["preco_total", "valor_total_item", "valor", "total", "sub_total"], None)
            quantidade = first_present(it, ["quantidade", "qtd", "quantidade_item", "qty"], 0)
            taxa = first_present(it, ["taxa_iva_percentagem", "taxa", "iva_percent", "iva_percentagem"], None)

            # converter numeros com segurança
            quantidade_f = safe_float(quantidade, 0.0)
            preco_unit_f = None if preco_unit is None else safe_float(preco_unit, 0.0)
            preco_total_f = None if preco_total is None else safe_float(preco_total, 0.0)

            # Heurística 1:
            # Se prec_unit is None or zero e existe preco_total e quantidade>0 -> derive unitário
            if (preco_unit_f is None or preco_unit_f == 0.0) and preco_total_f not in (None, 0.0) and quantidade_f > 0:
                # assumir preco_total pode conter IVA ou não; deixamos conversão para depois com taxa certa
                preco_unit_f = round(preco_total_f / quantidade_f, 2)

            # Heurística 2:
            # Detectar se preco_unit_f * quantidade == preco_total_f (com margem)
            # Se sim, e se tivermos taxa conhecida (ou taxa_padrao), verificar se preco_unit_f inclui IVA.
            taxa_f = None
            if taxa not in (None, ""):
                taxa_f = safe_float(taxa, 0.0)

            # se taxa não veio nos itens mas existe taxa_padrao do rodapé, usá-la
            if (taxa_f is None or taxa_f == 0.0) and taxa_padrao is not None:
                taxa_f = float(taxa_padrao)

            # Se temos preco_unit_f e preco_total_f e apenas o preco_total_f é o somatório com IVA,
            # podemos detectar inclusão de IVA: se preco_unit_f*quantidade_f ~= preco_total_f
            if preco_unit_f not in (None, 0.0) and preco_total_f not in (None, 0.0) and quantidade_f > 0:
                # comparação com tolerância pequena
                calc = round(preco_unit_f * quantidade_f, 2)
                if abs(calc - preco_total_f) < 0.01:
                    # pode ser que preco_unit_f já inclua IVA (ou não). Vamos checar com taxa_f:
                    if taxa_f and taxa_f > 0:
                        # Se preco_unit_f inclui IVA, então valor sem IVA = preco_unit_f / (1 + taxa/100)
                        # NOTA: Corrigi aqui para usar 'preco_unit_f' em vez de 'u_incl'
                        unit_sem_iva = round(preco_unit_f / (1.0 + taxa_f / 100.0), 2)
                        # recomputar
                        preco_unit_f = unit_sem_iva

            # Heurística 3:
            # Se não existia preco_total mas LLM colocou preco_unit já com IVA e rodapé total mostra total menor,
            # detectamos por comparação global mais abaixo (recalculo do total)
            # Formar o item normalizado
            normalized = {
                "descricao": descricao or "",
                "preco_unitario": float(f"{(preco_unit_f or 0.0):.2f}"),
                "quantidade": float(f"{quantidade_f:.6g}"),  # mantém precisão de quantidade (ex: 1, 1.5)
                "taxa_iva_percentagem": float(taxa_f or 0.0),
            }

            # também manter campos auxiliares se existirem
            if preco_total_f not in (None, 0.0):
                normalized["preco_total_extraido"] = float(f"{preco_total_f:.2f}")

            normalized_items.append(normalized)

        # Agora recalcular subtotal / iva / total a partir dos itens (considerando taxa em cada item)
        subtotal = 0.0
        iva_calc = 0.0
        # Primeiro pass: se algum item parece ter preço unitario que inclui IVA, tentar detectar pela soma
        for it in normalized_items:
            q = safe_float(it.get("quantidade", 0.0))
            u = safe_float(it.get("preco_unitario", 0.0))
            t_tax = safe_float(it.get("taxa_iva_percentagem", 0.0))
            subtotal += round(u * q, 2)
            iva_calc += round((u * q) * (t_tax / 100.0), 2)

        subtotal = round(subtotal, 2)
        iva_calc = round(iva_calc, 2)
        total_calc = round(subtotal + iva_calc, 2)

        # Se existir informação do rodapé (det), priorizamos onde fizer sentido e usamos-a para ajustar itens
        total_iva_footer = det.get("total_iva")
        total_com_iva_footer = det.get("total_com_iva")
        total_liquido_footer = det.get("total_liquido")

        # Se os totais dos itens não baterem com total_com_iva_footer, tentamos detectar se os preços unitários
        # estão com IVA incluído — neste caso, convertemos unitários para sem IVA usando a taxa (se homogénea).
        if total_com_iva_footer and abs(total_calc - float(total_com_iva_footer)) > 0.5:
            # calcular taxa homogénea se possível
            taxas_presentes = {round(safe_float(it.get("taxa_iva_percentagem", 0.0)), 6) for it in normalized_items}
            if len(taxas_presentes) == 1:
                taxa_hom = taxas_presentes.pop()
                if taxa_hom > 0:
                    # verificar se convertendo unitários (dividindo por 1+taxa) aproxima total
                    subtotal_candidate = 0.0
                    iva_candidate = 0.0
                    for it in normalized_items:
                        q = safe_float(it["quantidade"], 0.0)
                        u_incl = safe_float(it["preco_unitario"], 0.0)
                        u_excl = round(u_incl / (1.0 + taxa_hom / 100.0), 2)
                        subtotal_candidate += round(u_excl * q, 2)
                        iva_candidate += round((u_excl * q) * (taxa_hom / 100.0), 2)
                    total_candidate = round(subtotal_candidate + iva_candidate, 2)
                    # Se total_candidate estiver mais próximo do rodapé, fazemos a conversão
                    if abs(total_candidate - float(total_com_iva_footer)) < abs(total_calc - float(total_com_iva_footer)):
                        # aplicar conversão
                        for it in normalized_items:
                            u_incl = safe_float(it["preco_unitario"], 0.0)
                            u_excl = round(u_incl / (1.0 + taxa_hom / 100.0), 2)
                            it["preco_unitario"] = float(f"{u_excl:.2f}")
                        # recompute
                        subtotal = subtotal_candidate
                        iva_calc = iva_candidate
                        total_calc = total_candidate

        # Após ajuste, se itens tiverem taxa 0 e taxa_padrao detectada, aplicar taxa_padrao
        if taxa_padrao is not None:
            for it in normalized_items:
                if safe_float(it.get("taxa_iva_percentagem", 0.0), 0.0) == 0.0:
                    it["taxa_iva_percentagem"] = float(taxa_padrao)

        # Recalcular subtotal/iva/total final com taxas atualizadas
        subtotal = round(sum(safe_float(it["preco_unitario"], 0.0) * safe_float(it["quantidade"], 0.0) for it in normalized_items), 2)
        iva_calc = round(sum(
            (safe_float(it["preco_unitario"], 0.0) * safe_float(it["quantidade"], 0.0)) * (safe_float(it["taxa_iva_percentagem"], 0.0) / 100.0)
            for it in normalized_items
        ), 2)
        total_calc = round(subtotal + iva_calc, 2)

        # Agora priorizar valores do rodapé quando existirem
        if total_iva_footer and safe_float(total_iva_footer, 0.0) > 0:
            data["total_iva"] = float(f"{safe_float(total_iva_footer):.2f}")
        else:
            data["total_iva"] = float(f"{iva_calc:.2f}")

        if total_com_iva_footer and safe_float(total_com_iva_footer, 0.0) > 0:
            data["valor_total_documento"] = float(f"{safe_float(total_com_iva_footer):.2f}")
        else:
            if total_liquido_footer and safe_float(total_liquido_footer, 0.0) > 0 and data.get("total_iva") is not None:
                data["valor_total_documento"] = float(round(safe_float(total_liquido_footer) + safe_float(data["total_iva"]), 2))
            else:
                data["valor_total_documento"] = float(f"{total_calc:.2f}")

        # Valor pago: prioriza total_com_iva (rodapé) se existir
        if det.get("total_com_iva") is not None:
            data["valor_pago"] = float(f"{safe_float(det.get('total_com_iva')):.2f}")
        else:
            data["valor_pago"] = float(f"{data.get('valor_total_documento', 0.0):.2f}")

        # Normaliza itens finais (formatos e valores coerentes)
        for it in normalized_items:
            # Garantir tipos e formatação
            it["preco_unitario"] = float(f"{safe_float(it.get('preco_unitario', 0.0)):.2f}")
            # quantidade pode ser inteiro 1 ou decimal: limitar a 6 dígitos significativos
            it["quantidade"] = float(it.get("quantidade", 0.0))
            it["taxa_iva_percentagem"] = float(f"{safe_float(it.get('taxa_iva_percentagem', 0.0)):.2f}")
            # calcular preco_total com base em unitário (sem inventar: unitario * quantidade)
            it["preco_total_calculado"] = float(f"{round(it['preco_unitario'] * it['quantidade'], 2):.2f}")

        # Substituir items no data pela versão normalizada
        data["items"] = normalized_items

        # Normaliza monetários finais (segunda vez para garantir strings -> float)
        data["valor_total_documento"] = float(f"{safe_float(data.get('valor_total_documento', 0.0)):.2f}")
        data["total_iva"] = float(f"{safe_float(data.get('total_iva', 0.0)):.2f}")
        data["valor_pago"] = float(f"{safe_float(data.get('valor_pago', 0.0)):.2f}")

    except Exception as e:
        # Log do erro sem interromper fluxo
        print(f"[VALIDACAO] Erro: {e}")

    return data


# =========================
# Endpoint
# =========================
@app.post("/ocr")
async def ocr_and_structured_extract(
    file: UploadFile = File(...),
    # AQUI ESTÁ A MUDANÇA: company_id agora é do tipo 'int'
    company_id: int = Query(..., description="ID único da empresa para a qual o documento está a ser processado (número inteiro)."),
):
    print(f"Recebida requisição para company_id: {company_id} (tipo: {type(company_id)})")

    fname = (file.filename or "").lower()
    if not fname.endswith((".pdf", ".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Formato de arquivo não suportado. Envie PDF/PNG/JPG.")

    # A validação do company_id agora é tratada automaticamente pelo Pydantic/FastAPI
    # Se o valor não puder ser convertido para int, um erro 422 será retornado.

    try:
        data_bytes = await file.read()
        if not data_bytes:
            raise HTTPException(status_code=400, detail="Arquivo vazio.")

        if fname.endswith(".pdf"):
            extracted_text = extract_text_from_pdf_stream(data_bytes)
        else:
            extracted_text = extract_text_from_image_stream(data_bytes)

        if not extracted_text or not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Nenhum texto extraído do documento.")

        extracted_data = run_llm_structured_extraction(extracted_text)
        extracted_data = validar_e_corrigir_dados(extracted_data, extracted_text)

        # Retornar o company_id no resultado para confirmação
        return {
            "company_id": company_id, # O company_id aqui já será um int
            "extracted_text": extracted_text,
            "extracted_data": extracted_data
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Erro no processamento para company_id {company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")