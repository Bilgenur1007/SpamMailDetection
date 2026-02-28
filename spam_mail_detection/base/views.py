from django.shortcuts import render, redirect
from django.utils.translation import get_language
from django.utils import translation
from django.conf import settings
import os
import re
import pickle

# ─────────────────────────────────────────────
# Module-level model cache (performans: her request'te yeniden yüklenmez)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_model_cache = {}


def _load_ml(folder, model_name="lightgbm"):
    key = f"ml_{folder}_{model_name}"
    if key not in _model_cache:
        model_path = os.path.join(MODELS_DIR, folder, f"{model_name}.pkl")
        vec_path = os.path.join(MODELS_DIR, folder, "tfidf_vectorizer.pkl")
        _model_cache[key] = (
            pickle.load(open(model_path, "rb")),
            pickle.load(open(vec_path, "rb")),
        )
    return _model_cache[key]


def _load_dl(folder):
    key = f"dl_{folder}"
    if key not in _model_cache:
        from tensorflow.keras.models import load_model as keras_load
        model_path = os.path.join(MODELS_DIR, folder, "spam_classifier.keras")
        tok_path = os.path.join(MODELS_DIR, folder, "tokenizer.pickle")
        keras_model = keras_load(model_path)
        with open(tok_path, "rb") as f:
            tokenizer = pickle.load(f)
        _model_cache[key] = (keras_model, tokenizer)
    return _model_cache[key]


# ─────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────
def _rule_based(text):
    spam_kw = [
        "ödül", "hediye", "kazandınız", "tıklayın", "acele edin", "şimdi satın alın",
        "ücretsiz", "bedava", "hemen kazanın", "para kazanın", "kampanya",
        "free", "win", "cash", "prize", "offer", "click here", "buy now", "urgent",
        "limited time", "congratulations", "you have won", "act now", "guarantee", "deal",
    ]
    return int(any(kw in text.lower() for kw in spam_kw) or len(text.strip()) < 5)


def _is_turkish(text):
    return bool(re.search(r"[çğıöşüÇĞİÖŞÜ]", text))


def _predict_dl(text, tokenizer, model):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=150)
    return int(model.predict(padded, verbose=0)[0][0] > 0.5)


# ─────────────────────────────────────────────
# Ana tahmin fonksiyonu
# ─────────────────────────────────────────────
def getPrediction(mailTitle, mailContent, mailUrl, spamFilter, language):
    text_map = {
        "Mail Başlık": mailTitle,
        "Mail İçerik": mailContent,
        "Mail Url": mailUrl,
        "Bütün Mail": f"{mailTitle} {mailContent} {mailUrl}",
    }
    text = text_map.get(spamFilter)
    if text is None:
        return "no", []

    votes = {}
    used_models = ["rule_based_check (kural tabanlı)"]
    votes["rule_based"] = _rule_based(text)

    try:
        if _is_turkish(text) or language == "tr":
            # Türkçe metin → Naive Bayes (email_detection)
            model, vec = _load_ml("email_detection", "naive_bayes")
            votes["turkish_nb"] = int(model.predict(vec.transform([text]))[0])
            used_models.append("email_detection/naive_bayes.pkl (Türkçe NB)")
        else:
            # İngilizce metin → LightGBM + DL (email_detection) + LightGBM (ceas)
            model, vec = _load_ml("email_detection", "lightgbm")
            votes["email_ml"] = int(model.predict(vec.transform([text]))[0])
            used_models.append("email_detection/lightgbm.pkl (ML)")

            dl_model, tokenizer = _load_dl("email_detection")
            votes["email_dl"] = _predict_dl(text, tokenizer, dl_model)
            used_models.append("email_detection/spam_classifier.keras (DL)")

            ceas_model, ceas_vec = _load_ml("ceas", "lightgbm")
            votes["ceas_ml"] = int(ceas_model.predict(ceas_vec.transform([text]))[0])
            used_models.append("ceas/lightgbm.pkl (CEAS ML)")

    except Exception as e:
        print(f"[getPrediction] Model yükleme hatası: {e}")

    valid = [v for v in votes.values() if v is not None]
    is_spam = sum(valid) >= len(valid) / 2
    return ("yes" if is_spam else "no"), used_models


# ─────────────────────────────────────────────
# View'lar
# ─────────────────────────────────────────────
def home(request):
    return render(request, "index.html", {
        "LANGUAGE_CODE": get_language(),
    })


def result(request):
    if request.method != "POST":
        return redirect("home")

    language = request.session.get(settings.LANGUAGE_COOKIE_NAME, "en")
    mailTitle = request.POST.get("mailTitle", "")
    mailContent = request.POST.get("mailContent", "")
    mailUrl = request.POST.get("mailUrl", "")
    spamFilter = request.POST.get("spamFilter", "")

    result_text, used_models = getPrediction(mailTitle, mailContent, mailUrl, spamFilter, language)

    return render(request, "result.html", {
        "result": result_text,
        "language": language,
        "used_models": used_models,
    })


def set_language(request):
    if request.method == "POST":
        language = request.POST.get("LANGUAGE_CODE")
        if language:
            request.session[settings.LANGUAGE_COOKIE_NAME] = language
            translation.activate(language)
    return redirect(request.META.get("HTTP_REFERER", "/"))
