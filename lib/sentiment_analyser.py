# import packages in the function to prevent circular import error
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np


def softmax(x, axis=None):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)

    return " ".join(new_text)


def sentiment_analyser():

    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)

    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    #model.save_pretrained(MODEL)
    text = str(input("Enter text: "))
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Labels
    labels = ["Negative: ", "Neutral: ", "Positive: "]

    # Add labels to each element
    labeled_scores = [f"{label}{value}" for label, value in zip(labels, scores)]

    print(labeled_scores)

    '''
    # TF
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
