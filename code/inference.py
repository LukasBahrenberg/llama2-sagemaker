import torch
from transformers import AutoModel, AutoTokenizer

model_name = "model"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def model_fn(model_dir):
    model = AutoModel.from_pretrained(model_dir)
    return model

def input_fn(input_data, content_type):
    return tokenizer(input_data, return_tensors="pt")

def predict_fn(input_data, model):
    with torch.no_grad():
        return model(**input_data).logits.tolist()

def output_fn(predictions, content_type):
    return str(predictions)