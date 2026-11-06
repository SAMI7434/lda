from fastapi import FastAPI
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel
import nltk, re

nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI(title="LDA Topic Predictor")
lda_model = LdaModel.load("lda_model.model")
id2word = Dictionary.load("lda_dictionary.dict")

STOP = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

TOPIC_NAMES = {0: "Sports", 1: "Space", 2: "Graphics", 3: "Religion"}

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "LDA API is live! POST /predict with {'text': 'your sentence'}"}

@app.post("/predict")
def predict(input: TextInput):
    text = re.sub(r'\S*@\S*\s?', '', input.text)
    tokens = simple_preprocess(text, deacc=True, min_len=3)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP]
    bow = id2word.doc2bow(tokens)
    probs = lda_model.get_document_topics(bow, minimum_probability=0)
    best = max(probs, key=lambda x: x[1])
    words = [w for w, _ in lda_model.show_topic(best[0], topn=5)]
    return {
        "topic_id": best[0],
        "topic_name": TOPIC_NAMES.get(best[0], "Unknown"),
        "confidence": round(best[1], 3),
        "top_words": words
    }