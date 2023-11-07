from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from datetime import datetime
import torch


articles = ["Şeful ONU spune că nu există o soluţie militară în Siria", "Şeful ONU spune că nu există o soluţie militară în Siria", "Şeful ONU spune că nu există o soluţie militară în Siria", "Şeful ONU spune că nu există o soluţie militară în Siria"]

model_id = "facebook/nllb-200-3.3B"
# model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# inputs = tokenizer(article, return_tensors="pt")

# translated_tokens = model.generate(
#     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["spa_Latn"], max_length=30
# )
# print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])

print('cuda available:', torch.cuda.is_available())

print("Pipeline approach:")

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

generator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="ron_Latn",
    tgt_lang="spa_Latn",
    max_length=400,
    device=3,
)
start = datetime.now()
print("Translating...")
print(generator(articles))
end = datetime.now()
print(end-start)
