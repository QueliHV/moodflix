# test_translation.py
import torch
from transformers import (
    MarianMTModel, MarianTokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM
)

def try_marian_romance_en():
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

def try_unicamp_t5():
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

if __name__ == "__main__":
    # 1) tente Marian (ROMANCE->en); 2) fallback T5 da Unicamp
    for loader in (try_marian_romance_en, try_unicamp_t5):
        try:
            translate, used = loader()
            print(f"✅ Tradutor carregado: {used}")
            break
        except Exception as e:
            print(f"Falhou {loader.__name__}: {e}")
    else:
        raise SystemExit("Não foi possível carregar nenhum tradutor.")

    frases = [
        "estou cansada",
        "estou com medo",
        "estou muito feliz hoje",
        "estou apaixonada",
        "estou surpreso com a notícia",
        "estou triste porque meu time perdeu",
    ]
    for f in frases:
        en = translate(f)
        print(f"PT: {f}\nEN: {en}\n" + "-"*40)