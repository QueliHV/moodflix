# core/inference.py
from __future__ import annotations
from pathlib import Path
import joblib
import torch
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- Modelo SVM EN ---
_BASE = Path(__file__).resolve().parent.parent
_ART  = _BASE / "model" / "artifacts"
_MODEL_PATH = _ART / "emotion_svm_en_val_best.joblib"
clf_en = joblib.load(_MODEL_PATH)

# --- Tradutor PT->EN (singleton) ---
_translator_fn = None
_translator_name = None

# modelo de tradução 1
def _load_marian_romance_en():
    model_id = "Helsinki-NLP/opus-mt-ROMANCE-en"
    tok = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def translate(text: str, max_new_tokens=128):
        batch = tok([text], return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            gen = model.generate(**batch, num_beams=4, max_new_tokens=max_new_tokens)
        return tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
    return translate, model_id

# modelo de tradução 2
def _load_unicamp_t5():
    model_id = "unicamp-dl/translation-pt-en-t5"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def translate(text: str, max_new_tokens=128):
        prompt = f"translate Portuguese to English: {text}"
        batch = tok([prompt], return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            gen = model.generate(**batch, num_beams=4, max_new_tokens=max_new_tokens)
        return tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
    return translate, model_id

# Função de tradução
def _get_translator():
    global _translator_fn, _translator_name
    if _translator_fn is not None:
        return _translator_fn, _translator_name
    # tenta Marian → fallback T5
    try:
        _translator_fn, _translator_name = _load_marian_romance_en()
    except Exception:
        _translator_fn, _translator_name = _load_unicamp_t5()
    return _translator_fn, _translator_name


def translate_pt2en(text: str, max_new_tokens: int = 128) -> str:
    tr, _ = _get_translator()
    out = tr(text, max_new_tokens=max_new_tokens).strip()
    if not out or out.lower() == text.strip().lower():
        out = tr(text + ".", max_new_tokens=max_new_tokens).strip()
    return out


# =========================
# 1) Mapeamento emoção -> gêneros TMDb
# =========================
# Emoção → gêneros que reforçam
EMOTION_TO_GENRES_DIRECT = {
    "joy":      ["Comedy", "Family", "Animation"],
    "love":     ["Romance", "Drama"],
    "sadness":  ["Drama"],
    "anger":    ["Action", "Crime", "Thriller"],
    "fear":     ["Horror", "Mystery", "Thriller"],
    "surprise": ["Mystery", "Adventure", "Sci-Fi"],
}

# Emoção → gêneros que aliviam ou mudam
EMOTION_TO_GENRES_INVERSE = {
    "joy":      ["Drama"],               # traz profundidade
    "love":     ["Comedy"],              # leveza
    "sadness":  ["Comedy", "Family"],    # anima
    "anger":    ["Comedy", "Romance"],   # acalma
    "fear":     ["Animation", "Family"], # conforto
    "surprise": ["Romance", "Drama"],    # estabilidade
}

# =========================
# 2) Pipeline de inferência (texto em PT)
# =========================
def predict_emotion_pt(text_pt: str) -> tuple[str, str, dict[str, float]]:
    """
    Recebe texto em PT, traduz para EN, aplica o SVM (treinado em EN) e retorna:
        (label_predito, texto_en_usado_pelo_modelo, detalhes_de_scores)

    detalhes_de_scores: dict label->score (decision_function) ou prob (se disponível)
    """
    # 2.1) traduz PT->EN (com fallback mínimo para não quebrar fluxo)
    try:
        text_en = translate_pt2en(text_pt)
    except Exception:
        text_en = text_pt  # em caso de erro de tradução, ainda tentamos

    # 2.2) predição
    pred = clf_en.predict([text_en])[0]

    # 2.3) detalhamento (scores ou probabilidades)
    details: dict[str, float] = {}
    clf = clf_en.named_steps["clf"]
    tfidf = clf_en.named_steps["tfidf"]
    X = tfidf.transform([text_en])

    if hasattr(clf, "predict_proba"):
        # modelos com probabilidade nativa (ex.: LogisticRegression)
        probs = clf.predict_proba(X)[0]
        details = {c: round(float(p) * 100, 1) for c, p in zip(clf.classes_, probs)}
    elif hasattr(clf, "decision_function"):
        # SVM retorna "scores" não normalizados → aplicamos MinMaxScaler
        scores = clf.decision_function(X)
        if scores.ndim == 1:  # pode vir como vetor simples
            scores = [scores]
        raw = np.array(scores[0]).reshape(-1, 1)
        scaled = MinMaxScaler(feature_range=(0, 100)).fit_transform(raw).flatten()
        details = {c: round(float(s), 1) for c, s in zip(clf.classes_, scaled)}


    # 2.4) regra leve em PT (apenas se confiança baixa)
    lex = {
        "cansad": "sadness",
        "trist": "sadness",
        "feliz": "joy",
        "contente": "joy",
        "apaixonad": "love",
        "com raiva": "anger",
        "irritad": "anger",
        "assustad": "fear",
        "ansios": "fear",
        "medo": "fear",
        "surpres": "surprise",
    }
    low_pt = text_pt.lower()
    for key, forced in lex.items():
        if key in low_pt:
            if not details:
                pred = forced
            else:
                top2 = sorted(details.items(), key=lambda x: x[1], reverse=True)[:2]
                margin = (top2[0][1] - top2[1][1]) if len(top2) == 2 else 1.0
                if margin < 0.10:  # baixa separação -> aplica a regra
                    pred = forced
            break

    return pred, text_en, details