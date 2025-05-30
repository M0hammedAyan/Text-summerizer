from transformers import T5Tokenizer, T5ForConditionalGeneration
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
text = """  """ # Add your text """between this""" to summarize

input_text = "summarize: "+ text
input_ids = tokenizer.encode(input_text, return_tensors="pt",max_length=500,truncation=True)
summary_ids = model.generate(
    input_ids,
    max_length=300,
    num_beams=4,
    length_penalty=2.0,
    early_stopping=True)
summary = tokenizer.decode(summary_ids[0],skip_spcial_tokens=True)
print("Summary: ", summary)