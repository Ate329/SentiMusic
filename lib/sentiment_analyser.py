from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from lib.logging_config import config
import numpy as np


logger = config()


def softmax(x, axis=None):
    logger.info("Computing softmax values for the emotion scores...")

    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


# Preprocess text (username and link placeholders)
def preprocess(text):
    logger.info("Preprocessing the text...")

    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)

    return " ".join(new_text)


def sentiment_analyser():
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

    logger.info("Automatically tokenising the model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    logger.info("Reading the model config...")
    config = AutoConfig.from_pretrained(MODEL)

    # With pytoch
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # model.save_pretrained(MODEL)
    text = str(input("Enter text: "))
    text = preprocess(text)

    logger.info("Encoding the input...")
    encoded_input = tokenizer(text, return_tensors='pt')

    logger.info("Analysing the sentiments...")
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    logger.info("Success!")
    logger.info("Printing the scores...")

    # Labels
    labels = ["Negative: ", "Neutral: ", "Positive: "]

    # Add labels to each element
    labeled_scores = [f"{label}{value}" for label,
                      value in zip(labels, scores)]

    print(labeled_scores)

    '''
    # With tensorflow
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    text = "Covid cases are increasing fast!"
    encoded_input = tokenizer(text, return_tensors='tf')
    output = model(encoded_input)
    scores = output[0][0].numpy()
    scores = softmax(scores)
    '''

    # Print labels and scores
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        print(f"{i+1}) {l} {np.round(float(s), 4)}")

    # print(type(scores))
    return labeled_scores
