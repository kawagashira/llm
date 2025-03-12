#!/usr/bin/env python
from transformers import BertTokenizer, BertModel
import torch

# トークナイザーとモデルのロード
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# サンプル文章
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

# モデルでエンコーディング
with torch.no_grad():
    outputs = model(**inputs)

# 最終的な埋め込みベクトル
embeddings = outputs.last_hidden_state
print(embeddings.shape)  # (1, トークン数, 768)
print(embeddings)
