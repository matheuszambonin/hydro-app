"""
run_app.py
----------
Launcher do HydroPump para distribuição via PyInstaller (modo --windowed).

Responsabilidades:
  1. Rodar o servidor do Streamlit DENTRO do mesmo processo (sem subprocess.exe
     externo, sem janela de terminal), usando a API de bootstrap do Streamlit.
  2. Escolher uma porta livre da máquina automaticamente (evita conflito caso
     o usuário já tenha algo rodando na 8501).
  3. Abrir o navegador padrão do Windows assim que o servidor responder.
  4. Evitar múltiplas instâncias (se o usuário clicar duas vezes no atalho).

Este arquivo é o "entry point" referenciado no app.spec (Analysis -> ["run_app.py"]).
"""

import os
import sys
import socket
import threading
import time
import webbrowser
import urllib.request
import tempfile

APP_NAME = "HydroPump"


# --------------------------------------------------------------------------
# 0. Em builds --windowed (console=False) sys.stdout/stderr podem vir como
#    None. Várias libs (streamlit, matplotlib, etc.) fazem print()/logging
#    e isso quebra com "AttributeError: 'NoneType' object has no attribute
#    'write'". Redirecionamos para um arquivo de log em vez de para o vazio,
#    o que ajuda a depurar problemas relatados pelo usuário final.
# --------------------------------------------------------------------------
def _redirect_std_streams():
    log_dir = os.path.join(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()), APP_NAME)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "hydropump.log")
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file
    return log_path


# --------------------------------------------------------------------------
# 1. Utilitário para localizar recursos empacotados pelo PyInstaller
#    (sys._MEIPASS aponta para a pasta temporária/onedir onde os "datas"
#    declarados no app.spec foram extraídos).
# --------------------------------------------------------------------------
def resource_path(relative_path: str) -> str:
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------
# 2. Trava simples de instância única baseada em socket local.
#    Se o usuário clicar duas vezes no atalho da Área de Trabalho, a segunda
#    instância detecta a trava, apenas abre o navegador na mesma porta salva
#    e encerra, em vez de subir um segundo servidor.
# --------------------------------------------------------------------------
def _lock_dir():
    d = os.path.join(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()), APP_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def acquire_single_instance_lock():
    lock_path = os.path.join(_lock_dir(), "port.lock")
    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r") as f:
                existing_port = int(f.read().strip())
            url = f"http://localhost:{existing_port}"
            urllib.request.urlopen(url, timeout=1)
            # Servidor já está de pé: só abre o navegador e finaliza.
            webbrowser.open(url)
            sys.exit(0)
        except Exception:
            # Lock "morto" (processo anterior fechado sem limpar) -> segue normalmente.
            pass
    return lock_path


def write_lock(lock_path: str, port: int):
    with open(lock_path, "w") as f:
        f.write(str(port))


def release_lock(lock_path: str):
    try:
        os.remove(lock_path)
    except OSError:
        pass


# --------------------------------------------------------------------------
# 3. Abre o navegador assim que o servidor responder (evita a tela branca
#    "conexão recusada" que aparece se abrirmos o navegador cedo demais).
# --------------------------------------------------------------------------
def open_browser_when_ready(url: str, timeout: int = 30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=1)
            webbrowser.open(url)
            return
        except Exception:
            time.sleep(0.4)
    # Mesmo sem confirmar, tenta abrir no fim do timeout.
    webbrowser.open(url)


def main():
    _redirect_std_streams()

    lock_path = acquire_single_instance_lock()
    port = get_free_port()
    write_lock(lock_path, port)

    # Variáveis de ambiente do Streamlit: modo headless (sem abrir navegador
    # sozinho, quem controla isso é este script), sem telemetria/prompt de
    # e-mail, sem checagem de atualização.
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

    url = f"http://localhost:{port}"

    # app.py é copiado para dentro do pacote pelo app.spec (datas: "app.py" -> "app").
    app_script = resource_path(os.path.join("app", "app.py"))

    # Dispara em thread separada para não bloquear o bootstrap do Streamlit.
    threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()

    try:
        from streamlit.web import bootstrap

        flag_options = {
            "server.headless": True,
            "server.port": port,
            "server.address": "localhost",
            "browser.gatherUsageStats": False,
            "global.developmentMode": False,
        }
        # Assinatura estável desde Streamlit 1.12+: (main_script_path, is_hello, args, flag_options)
        bootstrap.run(app_script, False, [], flag_options)
    finally:
        release_lock(lock_path)


if __name__ == "__main__":
    main()
