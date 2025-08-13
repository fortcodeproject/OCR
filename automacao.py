import sys
import re
import unicodedata
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
# ... (código anterior)

    # Evita execução duplicada
    if not hasattr(globals(), "_execucao_itens_realizada"):
        globals()["_execucao_itens_realizada"] = False

    if globals()["_execucao_itens_realizada"]:
        print("[INFO] Preenchimento de itens já executado anteriormente. Ignorando repetição.")
    else:
        if data.get('items'):
            print("[INFO] Adicionando itens da fatura...")

            # DEBUG inicial: quantos select2 existem agora e seus textos
            try:
                selects_texts = await page.locator(
                    'div.form-group:has-text("Produto ou Serviço") .select2-selection'
                ).all_inner_texts()
                print(f"[DEBUG] Select2 encontrados (inicial): total={len(selects_texts)} -> {selects_texts}")
            except Exception as e:
                print(f"[WARN] Falha ao ler select2 iniciais: {e}")

            for i, item in enumerate(data['items']):
                descricao_limpa = (item.get('descricao') or '').strip()
                print(f"\n[INFO] Processando item {i + 1}: '{descricao_limpa}'")

                try:
                    # ===== 1) Localiza o Select2 relativo ao campo de preço =====
                    price_selector = f"input[wire\\:model\\.lazy='carts.{i}.price']"
                    price_count = await page.locator(price_selector).count()
                    print(f"[DEBUG] Price selector '{price_selector}' count = {price_count}")

                    product_select2_clicked = False
                    if price_count > 0:
                        handle = await page.evaluate_handle(
                            """(index) => {
                                const allInputs = Array.from(document.querySelectorAll('input'));
                                const price = allInputs.find(el => el.getAttribute && el.getAttribute('wire:model.lazy') === `carts.${index}.price`);
                                if (!price) return null;
                                let el = price;
                                for (let k = 0; k < 6; k++) {
                                    if (!el) break;
                                    const found = el.querySelector('.select2-selection, .select2-container, .select2-selection__rendered');
                                    if (found) return found;
                                    el = el.parentElement;
                                }
                                return document.querySelector('.select2-selection');
                            }""",
                            i
                        )
                        elem = handle.as_element() if handle else None
                        if elem:
                            await elem.click()
                            product_select2_clicked = True
                            print("[DEBUG] Select2 relativo ao preço clicado com sucesso.")
                        else:
                            print("[WARN] Nenhum Select2 relativo encontrado via preço.")

                    # ===== 2) Fallback global =====
                    if not product_select2_clicked:
                        selects = page.locator('div.form-group:has-text("Produto ou Serviço") .select2-selection')
                        total_selects = await selects.count()
                        print(f"[DEBUG] Select2 (fallback global) total = {total_selects}")
                        if total_selects == 0:
                            raise TimeoutError("Nenhum .select2-selection encontrado na página.")
                        idx_to_use = i if i < total_selects else total_selects - 1
                        await selects.nth(idx_to_use).scroll_into_view_if_needed()
                        await selects.nth(idx_to_use).click()
                        print(f"[DEBUG] Clique no Select2 global index {idx_to_use}.")
                        product_select2_clicked = True

                    # ===== 3) Campo de busca do Select2 =====
                    search_input_found = False
                    for sel in [
                        'span.select2-search.select2-search--dropdown > input',
                        'input.select2-search__field',
                        'body .select2-search input'
                    ]:
                        try:
                            await page.wait_for_selector(sel, state='visible', timeout=4000)
                            await page.fill(sel, descricao_limpa)
                            print(f"[DEBUG] Campo de busca visível ({sel}) e preenchido.")
                            search_input_found = True
                            break
                        except:
                            pass

                    if not search_input_found:
                        print(f"[ERRO] Campo de busca não apareceu para '{descricao_limpa}'.")
                        await page.screenshot(path=f"debug_item_{i+1}_no_search.png")
                        continue

                    # ===== 4) Opções de resultado =====
                    try:
                        await page.wait_for_selector("body ul.select2-results__options li", state="visible", timeout=6000)
                        options = await page.locator("body ul.select2-results__options li").all_inner_texts()
                    except:
                        options = []
                    print(f"[DEBUG] Opções para item {i+1}: {options}")

                    if not options:
                        raise TimeoutError(f"Nenhuma opção apareceu para '{descricao_limpa}'.")

                    # ===== 5) Seleção =====
                    termo_regex = re.escape(descricao_limpa)
                    result_selector = f'body ul.select2-results__options li:text-matches(".*{termo_regex}.*", "i")'
                    if await page.locator(result_selector).count() > 0:
                        await page.click(result_selector)
                        print(f"[INFO] Produto '{descricao_limpa}' selecionado (exato).")
                    else:
                        normalized = unicodedata.normalize('NFD', descricao_limpa)
                        normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
                        normalized = re.sub(r'[^0-9A-Za-z ]', ' ', normalized).strip()
                        words = normalized.split()
                        term_candidates = [" ".join(words[:2]), words[0]] if words else []

                        selected = False
                        for t in term_candidates:
                            if not t:
                                continue
                            sel = f'body ul.select2-results__options li:text-matches(".*{re.escape(t)}.*", "i")'
                            if await page.locator(sel).count() > 0:
                                await page.click(sel)
                                print(f"[INFO] Produto '{descricao_limpa}' selecionado (parcial: '{t}').")
                                selected = True
                                break

                        if not selected:
                            await page.locator("body ul.select2-results__options li").nth(0).click()
                            print(f"[WARN] Seleção por fallback: primeira opção visível.")

                    # ===== 6) Preencher preço, quantidade, taxa =====
                    await page.locator(f"input[wire\\:model\\.lazy='carts.{i}.price']").fill(str(item.get('preco_unitario', 0)))
                    await page.locator(f"input[wire\\:model\\.lazy='carts.{i}.qtd']").fill(str(item.get('quantidade', 0)))
                    await page.locator(f"input[wire\\:model\\.lazy='carts.{i}.tax']").fill(str(item.get('taxa_iva_percentagem', 0)))
                    print(f"[INFO] Campos do item '{descricao_limpa}' preenchidos.")

                except TimeoutError as te:
                    print(f"[ERRO] Produto '{descricao_limpa}' não encontrado: {te}")
                    await page.screenshot(path=f"debug_item_{i+1}_timeout.png")
                except Exception as e:
                    print(f"[ERRO] Falha inesperada no item {i+1}: {e}")
                    await page.screenshot(path=f"debug_item_{i+1}_exception.png")

            globals()["_execucao_itens_realizada"] = True
        else:
            print("[AVISO] Nenhum item extraído do documento.")

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

            # Aguarda 20 minutos antes de fechar o navegador
            print("⏳ Esperando 20 minutos antes de fechar o navegador...")
            await asyncio.sleep(20 * 60)  # 20 minutos

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