"""
Pacote principal do MoodFlix 🎬

Este arquivo transforma a pasta `app/` em um pacote Python
e inicializa configurações globais (como variáveis do .env).
"""

import os
from dotenv import load_dotenv

# 1) Carrega variáveis de ambiente do arquivo .env automaticamente
load_dotenv()

# 2) Importa funções úteis do cliente TMDb e as expõe direto no pacote
from .tmdb_client import discover_movies, poster_url

# 3) Define o que fica público ao importar `from app import *`
__all__ = ["discover_movies", "poster_url"]