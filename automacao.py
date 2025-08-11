import sys
from datetime import datetime
import asyncio
from playwright.async_api import async_playwright, TimeoutError
import os
import json
from dotenv import load_dotenv

load_dotenv()

# URL da sua API FastAPI e da plataforma web
API_URL = os.getenv("API_URL")
LOGIN_URL = os.getenv("LOGIN_URL")
FORM_URL = os.getenv("FORM_URL")

# CREDENCIAIS DE LOGIN
USER_EMAIL = os.getenv("USER_EMAIL")
USER_PASSWORD = os.getenv("USER_PASSWORD")

async def perform_login(page):
    """Realiza o login na plataforma web."""
    print("Iniciando login...")
    await page.goto(LOGIN_URL)
    await page.wait_for_load_state('networkidle')

    try:
        await page.fill('input[placeholder="E-mail "]', USER_EMAIL)
        await page.fill('input[placeholder="Senha de Acesso"]', USER_PASSWORD)
        await page.click('button:has-text("Entrar")')
        await page.wait_for_url(lambda url: url != LOGIN_URL, timeout=60000)
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
            # Continua o script mesmo se o fornecedor não for encontrado

    if data.get('items'):
        print("Adicionando itens da fatura...")

        for i, item in enumerate(data['items']):
            print(f"Selecionando produto '{item.get('descricao', '')}'...")

            try:
                # Clicar no campo do produto (Select2)
                product_select2_container = page.locator('div.form-group:has-text("Produto ou Serviço") .select2-selection')
                await product_select2_container.click()

                # Digitar e selecionar o produto
                search_input_selector = 'span.select2-search.select2-search--dropdown > input'
                await page.wait_for_selector(search_input_selector, state='visible', timeout=10000)
                await page.fill(search_input_selector, item['descricao'])

                result_selector = f'ul.select2-results__options li:has-text("{item["descricao"]}")'
                await page.wait_for_selector(result_selector, state='visible', timeout=10000)
                await page.click(result_selector)

                # Após selecionar o produto, esperar o botão aparecer
                add_item_selector = 'button[wire\\:click="addItem"]'
                await page.wait_for_selector(add_item_selector, timeout=10000)
                await page.locator(add_item_selector).click()
                await page.wait_for_timeout(500)

                print(f"Preenchendo item {i + 1}: '{item.get('descricao', '')}'")

                # Preencher os campos
                await page.locator(f"input[wire\\:model\\.lazy='carts.{i}.price']").fill(str(item.get('preco_unitario', 0)))
                await page.locator(f"input[wire\\:model\\.lazy='carts.{i}.qtd']").fill(str(item.get('quantidade', 0)))
                await page.locator(f"input[wire\\:model\\.lazy='carts.{i}.tax']").fill(str(item.get('taxa_iva_percentagem', 0)))

                print(f"Item '{item.get('descricao', '')}' preenchido com sucesso.")

            except TimeoutError:
                print(f"Erro: Não foi possível selecionar o produto '{item.get('descricao', '')}'. Verifique se ele já existe na base de dados.")
                continue
            except Exception as e:
                print(f"Erro inesperado ao preencher o item {i+1}: {e}")
                continue

    # Preenchimento dos campos de pagamento
    payment_index = 0

    if data.get('valor_pago') is not None:
        await page.locator(f"input[wire\\:model='payments.{payment_index}.amount']").fill(str(data['valor_pago']))
        print(f"Valor pago preenchido: {data['valor_pago']}")

    if data.get('data_emissao'):
        try:
            raw_date = data['data_emissao']
            if '-' in raw_date:
                date_obj = datetime.strptime(raw_date, "%d-%m-%Y")
                formatted_date = date_obj.strftime("%Y-%m-%d")
            else:
                formatted_date = raw_date
            print(f"Data formatada: {formatted_date}")
            await page.locator(f"input[wire\\:model='payments.{payment_index}.payment_date']").fill(formatted_date)
        except ValueError:
            print(f"Erro ao converter data: {data['data_emissao']}")

    await page.wait_for_timeout(1000)

    try:
        await page.locator(f"select[wire\\:model='payments.{payment_index}.payment_method']").select_option('transferencia')
        print("Método de pagamento selecionado: Transferência")
    except Exception as e:
        print(f"Aviso: Não foi possível selecionar o método de pagamento. Erro: {e}")
    
    await page.locator(f"input[wire\\:model='payments.{payment_index}.invoice_file']").set_input_files(file_path)
    print("Comprovativo carregado.")

    await page.wait_for_timeout(1000)

    try:
        print("Tentando clicar no botão 'Criar Rascunho'...")
        await page.wait_for_selector('button[wire\\:target="save(1)"]', timeout=15000)
        await page.click('button[wire\\:target="save(1)"]')
        print("Botão 'Criar Rascunho' clicado com sucesso! Aguardando fechamento do modal...")
        await page.wait_for_selector('.modal-footer', state='hidden', timeout=3600000)
        print("Modal fechado com sucesso. Rascunho criado!")

    except TimeoutError:
        print("Aviso: O botão 'Criar Rascunho' não foi encontrado ou demorou demais.")
    except Exception as e:
        print(f"Erro ao tentar clicar no botão 'Criar Rascunho': {e}")

    print("\n✅ Formulário preenchido e rascunho criado com sucesso.")


async def run_automation(data, file_path: str):
    """Função principal que orquestra todo o processo."""
    browser = None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False, slow_mo=500)
            page = await browser.new_page()
            
            await perform_login(page)
            await fill_cost_form(page, data, file_path)

    except Exception as e:
        print(f"Ocorreu um erro no script de automação: {e}")
    finally:
        if browser and browser.is_connected():
            print("Fechando o navegador.")
            await browser.close()
            os.remove(file_path)
            print(f"Ficheiro temporário {file_path} removido.")

def main(file_path, extracted_data_str):
    """Ponto de entrada para o script."""
    try:
        extracted_data = json.loads(extracted_data_str)
        asyncio.run(run_automation(extracted_data, file_path))
    except json.JSONDecodeError:
        print("Erro: Dados JSON inválidos recebidos.")
    except Exception as e:
        print(f"Erro ao executar a automação: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python automacao.py <caminho_do_ficheiro> <dados_json>")
        sys.exit(1)
        
    main(sys.argv[1], sys.argv[2])