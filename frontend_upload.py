import streamlit as st
import requests
import os
import subprocess
import json
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do .env
load_dotenv()

# Obtém a URL da API, garantindo que o valor seja válido
API_URL = os.getenv("API_URL")
if not API_URL:
    st.error("A variável de ambiente API_URL não está definida.")
    st.stop()

# Configuração da página Streamlit
st.set_page_config(page_title="OCR Automático", layout="centered")
st.title("📄 Enviar Fatura para Processamento Automático")
st.markdown("Faça o upload de uma fatura para extrair os dados e iniciar a automação.")

# Widget de upload de ficheiro
uploaded_file = st.file_uploader("Selecione um ficheiro PDF, JPG ou PNG", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.button("Iniciar Extração e Preenchimento"):
        # Crie a pasta temporária para salvar o ficheiro
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)

        # Salva o ficheiro carregado em disco
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info(f"Ficheiro '{uploaded_file.name}' salvo temporariamente.")

        with st.spinner("Processando... Por favor, aguarde."):
            try:
                # Envia o ficheiro salvo para a API de OCR
                with open(file_path, 'rb') as f:
                    files = {'file': (uploaded_file.name, f, uploaded_file.type)}
                    response = requests.post(f"{API_URL}/ocr", files=files)
                    response.raise_for_status()
                
                # Obtém os dados extraídos da resposta da API
                extracted_data = response.json().get('extracted_data', {})
                if not extracted_data:
                    raise ValueError("A API não retornou dados de extração válidos.")
                
                st.success("✅ Dados extraídos com sucesso!")
                
                # Exibe os dados extraídos para o utilizador
                with st.expander("Ver dados extraídos", expanded=False):
                    st.json(extracted_data)

                # Converte os dados para uma string JSON para passar como argumento
                extracted_data_str = json.dumps(extracted_data)
                
                # Inicia o script de automação em um processo separado (não bloqueia a UI)
                # `subprocess.Popen` inicia o processo e continua imediatamente
                subprocess.Popen(["python", "automacao.py", file_path, extracted_data_str])

                st.success("🎉 A automação web foi iniciada com sucesso em segundo plano!")
                st.markdown("O navegador será aberto automaticamente para preencher o formulário. Pode continuar a usar esta página.")
                
            except requests.exceptions.RequestException as e:
                st.error(f"Erro ao conectar com a API: {e}")
            except ValueError as e:
                st.error(f"Erro nos dados da API: {e}")
            except Exception as e:
                st.error(f"Erro inesperado durante o processamento: {e}")