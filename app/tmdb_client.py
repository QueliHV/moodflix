import os
import requests

# 2) Guardar a chave e a URL base do TMDb
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"


if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY não encontrada. Verifique o arquivo .env")

# 3) Função utilitária para montar a URL do cartaz (pôster)
def poster_url(path, size="w342"):
    """
    Recebe o 'poster_path' retornado pela API
    e monta a URL completa da imagem.

    size pode ser: w185, w342, w500, original
    """
    return f"https://image.tmdb.org/t/p/{size}{path}" if path else None


# 4) Função principal: descobrir filmes por gêneros
def discover_movies(genres, language="pt-BR", 
                    page=1, sort_by="popularity.desc", 
                    min_vote_count=100, year_min=None, year_max=None):    

    """
    Busca filmes no TMDb de acordo com os gêneros.
    - genres: lista de nomes de gêneros (ex.: ["Comedy","Drama"])
    - language: idioma da resposta ("pt-BR")
    - page: página de resultados (a API retorna 20 filmes por página)
    - sort_by: conforme seleção do usário em tela
    - min_vote_count: número mínimo de votos (evita filmes obscuros)
    - Year_min e year_max: para filtrar classicos

    Retorna: lista de dicionários, cada um representando um filme.
    """

    # 4.1) Dicionário fixo que mapeia nome -> ID oficial no TMDb
    GENRE_NAME_TO_ID = {
        "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35, "Crime": 80,
        "Documentary": 99, "Drama": 18, "Family": 10751, "Fantasy": 14,
        "History": 36, "Horror": 27, "Music": 10402, "Mystery": 9648,
        "Romance": 10749, "Science Fiction": 878, "Thriller": 53,
        "War": 10752, "Western": 37
    }

    # 4.2) Converter lista de nomes ["Comedy","Drama"] em IDs ["35","18"]
    ids = [str(GENRE_NAME_TO_ID[g]) for g in genres if g in GENRE_NAME_TO_ID]

    # 4.3) Montar os parâmetros da requisição
    params = {
        "api_key": TMDB_API_KEY,        # sua chave secreta
        "language": language,           # idioma da resposta
        "sort_by": sort_by,
        "include_adult": "false",       # não trazer filmes +18
        "vote_count.gte": min_vote_count, # mínimo de votos para evitar filmes obscuros
        "page": page,                   # qual página trazer (20 por página)
    }

    if ids:
        params["with_genres"] = ",".join(ids)
    if year_min:
        params["primary_release_date.gte"] = f"{year_min}-01-01"
    if year_max:
        params["primary_release_date.lte"] = f"{year_max}-12-31"

    # 4.4) Faz a requisição GET para o endpoint /discover/movie
    r = requests.get(f"{TMDB_BASE}/discover/movie", params=params, timeout=20)

    # 4.5) Se algo deu errado, gera erro (ex.: chave inválida)
    r.raise_for_status()

    # 4.6) Retorna a lista de resultados
    return r.json().get("results") or []