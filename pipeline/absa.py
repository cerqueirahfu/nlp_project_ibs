from functools import lru_cache
from pathlib import Path
from setfit import AbsaModel

ASPECT_MODEL_ID = "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect"
POLARITY_MODEL_ID = "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity"
SPACY_MODEL = "en_core_web_sm"

MODEL_CACHE_DIR = Path("models/setfit_absa")
LOCAL_POLARITY_PATH = Path("models/setfit_absa_polarity_finetuned")


@lru_cache(maxsize=1)
def load_model() -> AbsaModel:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    polarity = (
        str(LOCAL_POLARITY_PATH) if LOCAL_POLARITY_PATH.exists() else POLARITY_MODEL_ID
    )
    return AbsaModel.from_pretrained(
        ASPECT_MODEL_ID,
        polarity,
        spacy_model=SPACY_MODEL,
        cache_dir=str(MODEL_CACHE_DIR),
    )


def predict(sentences):
    if not sentences:
        return []

    model = load_model()
    return model.predict(sentences)
