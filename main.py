from transformers import AutoModel, AutoTokenizer

model_name = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("Hello world!", return_tensors="pt")

outputs = model(**inputs)
