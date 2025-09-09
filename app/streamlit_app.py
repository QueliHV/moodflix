# streamlit_app.py
import base64
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.inference import (
    predict_emotion_pt,
    EMOTION_TO_GENRES_DIRECT,
    EMOTION_TO_GENRES_INVERSE,
)
from app.tmdb_client import discover_movies, poster_url

# ------------------ WARM-UP ------------------
if "warmed_up" not in st.session_state:
    st.info("🕐 Carregando modelos pela primeira vez (pode levar alguns segundos)...")
    # Chama o modelo 1x com uma frase dummy só para carregar o tradutor + SVM
    try:
        _ = predict_emotion_pt("teste de inicialização")
    except Exception as e:
        st.warning(f"Falha no warm-up: {e}")
    st.session_state["warmed_up"] = True
# ---------------------------------------------

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="MoodFlix", page_icon="🎬", layout="wide")

# ---------------------- CABEÇALHO -------------------
st.title("🎬 MoodFlix")
st.subheader("Recomenda filmes conforme seu humor")
st.markdown(
    "Descreva como você está se sentindo ou **como gostaria de se sentir** e eu vou sugerir filmes. "
    "_(Modelo próprio para classificar emoção + TMDb para buscar os filmes.)_"
)

# ---------------------- ENTRADA ---------------------
user_text = st.text_input(
    "Como você está se sentindo hoje?",
    placeholder="ex.: estou cansada, quero algo leve para ver com a família",
)

mode = st.radio("Quer um filme que…", ["Combinar", "Mudar"])
ordenacao = st.radio("Como ordenar?", ["Populares", "Nota alta", "Clássicos"])
top_k = st.slider("Quantas sugestões mostrar?", 3, 10, 5)

if st.button("Sugerir filmes"):
    if not user_text.strip():
        st.warning("Digite seu humor antes de pedir sugestões.")
        st.stop()

    # 1) Predizer emoção
    with st.spinner("Analisando seu humor..."):
        try:
            emotion, text_en, details = predict_emotion_pt(user_text)
        except Exception as e:
            st.error(f"Erro ao rodar o modelo: {e}")
            st.stop()

    # 2) Mapear emoção -> gêneros
    if mode == "Combinar":
        genres = EMOTION_TO_GENRES_DIRECT.get(emotion, ["Drama"])
    else:
        genres = EMOTION_TO_GENRES_INVERSE.get(emotion, ["Comedy"])

    # Info técnica 
    with st.expander("Informações técnicas Modelo"):
        st.write(f"**Texto em inglês usado pelo modelo:** {text_en}")
        st.write(f"**Modo:** `{mode}`")
        if details:
            st.markdown("### Percentual de confiança")

            # dicionário auxiliar
            EMOJI_MAP = {
                "joy": "🎉",
                "love": "❤️",
                "sadness": "😢",
                "anger": "😡",
                "fear": "😱",
                "surprise": "😮",}

            for emo, perc in sorted(details.items(), key=lambda x: x[1], reverse=True):
                emoji = EMOJI_MAP.get(emo, "")
                st.markdown(f"- {emoji} **{emo.capitalize()}** → {perc}%")

        st.markdown(f"**Emoção detectada:** `{emotion}`")
        st.caption(f"Gêneros mapeados: {', '.join(genres)}")   
    

    # 3) Buscar filmes (apenas UMA chamada)
    with st.spinner("Buscando filmes na TMDb..."):
        try:
            if ordenacao == "Populares":
                movies = discover_movies(
                    genres=genres, sort_by="popularity.desc", min_vote_count=100
                )
            elif ordenacao == "Nota alta":
                movies = discover_movies(
                    genres=genres, sort_by="vote_average.desc", min_vote_count=1000
                )
            else:  # Clássicos
                movies = discover_movies(
                    genres=genres, sort_by="vote_average.desc", min_vote_count=1000, year_max=2010
                )

        except Exception as e:
            st.error(f"Erro ao consultar a TMDb: {e}")
            movies = []

    # 4) Exibir
    st.subheader("Sugestões")
    if not movies:
        st.info("Nenhum filme encontrado. Tente descrever um pouco mais como você está se sentindo.")
    else:
        items = movies[:top_k]
        cols = st.columns(len(items)) if items else [st]
        for i, it in enumerate(items):
            title = it.get("title") or it.get("name")
            year = (it.get("release_date") or "????")[:4]
            poster = poster_url(it.get("poster_path"))
            vote = it.get("vote_average", 0.0)
            overview = it.get("overview", "")

            with cols[i % len(cols)]:
                if poster:
                    st.image(poster, use_container_width=True)
                st.markdown(f"**{title}** ({year})")
                st.write(f"⭐ {vote}")
                if overview:
                    st.caption(overview)
else:
    st.info("Digite seu humor e clique em **Sugerir filmes**.")

# ---------------------- RODAPÉ ----------------------
st.markdown("---")
logo_path = Path(__file__).parent / "assets" / "tmdb_powered_blue.svg"

def img_tag(path: Path, width=180):
    mime = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:{mime};base64,{b64}" width="{width}" style="display:block;margin:10px auto;">'

html_logo = img_tag(logo_path, 180) if logo_path.exists() else '<span style="display:block;height:10px;"></span>'
st.markdown(
    f"""
    <div style="text-align:center;">
        {html_logo}
        <p style="font-size:14px; color: gray;">
            Este produto usa a API do TMDb.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)