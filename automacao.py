import requests
from datetime import datetime
import asyncio
from playwright.async_api import async_playwright, TimeoutError
import os

# URL da sua API FastAPI e da plataforma web
API_URL = os.getenv("API_URL", "http://localhost:8000")
LOGIN_URL = "https://fortcodedev.com/login"
FORM_URL = "https://fortcodedev.com/custos"

# CREDENCIAIS DE LOGIN (substitua pelas suas)
USER_EMAIL = os.getenv("USER_EMAIL", "geral@gmail.com")
USER_PASSWORD = os.getenv("USER_PASSWORD", "geral@gmail.com")

async def perform_login(page):
    """Realiza o login na plataforma web."""
    print("Iniciando login...")
    await page.goto(LOGIN_URL)
    await page.wait_for_load_state('networkidle')

    try:
        await page.fill('input[placeholder="E-mail "]', USER_EMAIL)
        await page.fill('input[placeholder="Senha de Acesso"]', USER_PASSWORD)
        await page.click('button:has-text("Entrar")')
        await page.wait_for_url(lambda url: url != LOGIN_URL, timeout=30000)
        print("Login bem-sucedido!")
    except Exception as e:
        raise Exception(f"Falha no login: {e}")


async def fill_cost_form(page, data, file_path):
    """Navega para a aba de custos e preenche o formulário."""
    print("Navegando para a aba 'Meus Custos'...")
    await page.goto(FORM_URL)
    await page.wait_for_load_state('networkidle')

    print("Esperando o botão 'Novo' ficar visível...")
    await page.wait_for_selector('button:has-text("Novo")', timeout=30000)

    print("Clicando no botão 'Novo' para abrir o modal...")
    await page.click('button:has-text("Novo")')

    print("Esperando que o formulário de registro de custos fique visível...")
    await page.wait_for_selector('input[name="invoice_number"]', state='visible', timeout=30000)
    print("Formulário de registro encontrado! Iniciando preenchimento...")

    # Preenchimento dos campos do modal
    if data.get('invoice_number'):
        await page.fill('input[name="invoice_number"]', data['invoice_number'])
        print(f"Número da fatura preenchido: {data['invoice_number']}")

    if data.get('supplier_name'):
        print("Preenchendo o campo do fornecedor (Select2)...")
        try:
            supplier_select2_container = page.locator('div.form-group.col-md-4:has-text("Fornecedor/Cliente")').locator('.select2-selection')
            await supplier_select2_container.click()

            search_input_selector = 'span.select2-search.select2-search--dropdown > input'
            await page.wait_for_selector(search_input_selector, state='visible', timeout=10000)
            await page.fill(search_input_selector, data['supplier_name'])

            result_selector = f'ul.select2-results__options li:has-text("{data["supplier_name"]}")'
            await page.wait_for_selector(result_selector, state='visible', timeout=10000)
            await page.click(result_selector)
            print(f"Fornecedor '{data['supplier_name']}' encontrado e selecionado.")
        except TimeoutError:
            print(f"Erro: Não foi possível selecionar o fornecedor '{data['supplier_name']}'. Verifique se ele já existe na sua base de dados com o nome exato.")

    if data.get('items'):
        print("Adicionando itens da fatura...")
        add_item_button = page.locator('button[wire:click="addItem"]')

        for i, item in enumerate(data['items']):
            if i > 0:
                await add_item_button.click()
                await page.wait_for_timeout(500)  # Pequeno delay entre adições

            print(f"Preenchendo item {i + 1}: '{item['descricao']}'")
            try:
                # Abre o Select2 do produto
                product_select2_container = page.locator(f'div.row:nth-of-type({i+1}) div.form-group.col-md-4:has-text("Produto ou Serviço") .select2-selection')
                await product_select2_container.click()

                # Espera e preenche o campo de busca do Select2
                search_input_selector = 'span.select2-search.select2-search--dropdown > input'
                await page.wait_for_selector(search_input_selector, state='visible', timeout=10000)
                await page.fill(search_input_selector, item['descricao'])

                # Seleciona a opção correta
                result_selector = f'ul.select2-results__options li:has-text("{item["descricao"]}")'
                await page.wait_for_selector(result_selector, state='visible', timeout=10000)
                await page.click(result_selector)

                # Preenche os campos de preço, quantidade e taxa de IVA
                await page.locator(f"input[wire\\:model\\.lazy='carts.{i}.price']").fill(str(item['preco_unitario']))
                await page.locator(f"input[wire\\:model\\.lazy='carts.{i}.qtd']").fill(str(item['quantidade']))
                await page.locator(f"input[wire\\:model\\.lazy='carts.{i}.tax']").fill(str(item['taxa_iva_percentagem']))
                
                print(f"Item '{item['descricao']}' preenchido com sucesso.")

            except TimeoutError:
                print(f"Erro: Não foi possível selecionar o produto '{item['descricao']}'. Verifique se ele já existe na base de dados com o nome exato.")
                continue
            except Exception as e:
                print(f"Erro inesperado ao preencher o item {i}: {e}")
                continue


    # Preenchimento dos campos de pagamento
    payment_index = 0

    if data.get('valor_pago') is not None:
        await page.locator(f"input[wire\\:model='payments.{payment_index}.amount']").fill(str(data['valor_pago']))
        print(f"Valor pago preenchido: {data['valor_pago']}")

    if data.get('data_emissao'):
        try:
            # Converter data se estiver no formato dd-mm-aaaa
            raw_date = data['data_emissao']
            if '-' in raw_date:
                date_obj = datetime.strptime(raw_date, "%d-%m-%Y")
                formatted_date = date_obj.strftime("%Y-%m-%d")
            else:
                formatted_date = raw_date  # assume que já está no formato correto
            print(f"Data formatada: {formatted_date}")
            await page.locator(f"input[wire\\:model='payments.{payment_index}.payment_date']").fill(formatted_date)
        except ValueError:
            print(f"Erro ao converter data: {data['data_emissao']}")

    # Adicionando um pequeno tempo de espera para garantir que o dropdown seja preenchido
    await page.wait_for_timeout(1000)

    try:
        await page.locator(f"select[wire\\:model='payments.{payment_index}.payment_method']").select_option('transferencia')
        print("Método de pagamento selecionado: Transferência")
    except Exception as e:
        print(f"Aviso: Não foi possível selecionar o método de pagamento. Erro: {e}")
    
    await page.locator(f"input[wire\\:model='payments.{payment_index}.invoice_file']").set_input_files(file_path)
    print("Comprovativo carregado.")

    #print("\nPreenchimento finalizado. O utilizador deve agora verificar e salvar o formulário manualmente.")
    try:
        print("Tentando clicar no botão 'Salvar'...")
        
        await page.wait_for_selector('button.btn-website[wire\\:target="save"]', timeout=15000)
        await page.click('button.btn-website[wire\\:target="save"]')

        print("Botão 'Salvar' clicado com sucesso!")

        await page.wait_for_selector('.modal-footer', state='hidden', timeout=30000)
        print("Modal de registro de custos fechado com sucesso!")

    except TimeoutError:
        print("Aviso: O botão 'Salvar' não foi encontrado ou a página demorou muito para responder.")
    except Exception as e:
        print(f"Erro ao tentar salvar o formulário: {e}")

    
    print("\nProcesso de preenchimento e salvamento finalizado.")

async def fill_form_with_ocr_data(file_path: str):
    """Função principal que orquestra todo o processo."""
    browser = None
    try:
        print(f"Enviando arquivo {file_path} para a API de OCR...")
        with open(file_path, "rb") as f:
            files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
            response = requests.post(f"{API_URL}/ocr", files=files)
            response.raise_for_status()

        data = response.json()['extracted_data']
        print("Dados extraídos da API:", data)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False, slow_mo=500)
            page = await browser.new_page()

            await perform_login(page)

            await fill_cost_form(page, data, file_path)

            print("Navegador permanecerá aberto para inspeção. Feche-o manualmente.")
            await page.wait_for_timeout(3600000)

    except requests.exceptions.RequestException as e:
        print(f"Erro ao conectar ou receber dados da sua API: {e}")
    except Exception as e:
        print(f"Ocorreu um erro no script de automação: {e}")
    finally:
        if browser and browser.is_connected():
            #pass
            print("Fechando o navegador.")
            await browser.close()

if __name__ == "__main__":
    file_path = r"C:\Users\IA Developer(Mario)\Downloads\2025-07 11 FACTURA AUTO BOULOS LDA.pdf"
    asyncio.run(fill_form_with_ocr_data(file_path))