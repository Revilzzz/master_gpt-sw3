from transformers import AutoTokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset, concatenate_datasets
import torch
import json

# Sökvägar
train_path = 'output_text_file.txt'  
qa_dataset_path = 'qa_dataset.json'  
model_name = "AI-Sweden-Models/gpt-sw3-356m-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

if "gpt2" in model_name.lower():
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    
    if 'text' in examples:
        
        texts = examples["text"]
    elif 'context' in examples and 'question' in examples and 'answer' in examples:
        
        texts = [f"Kontext: {ctxt} Fråga: {q} Svar: {a}" for ctxt, q, a in zip(examples['context'], examples['question'], examples['answer'])]
    else:
        
        raise ValueError("Okänt dataformat")

    
    model_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=512)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs



raw_dataset = load_dataset('text', data_files={'train': train_path}, split='train')
tokenized_raw_datasets = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Ladda och förbered QA-dataset
with open(qa_dataset_path, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

# Skapa en lista av strängar istället för en lista av dictionaries
qa_texts = [f"Kontext: {item['context']} Fråga: {item['question']} Svar: {item['answer']}" for item in qa_data]

# Skapa en Dataset från listan av strängar
qa_dataset = Dataset.from_dict({'text': qa_texts})
tokenized_qa_datasets = qa_dataset.map(tokenize_function, batched=True)

# Kombinera datasets
combined_datasets = concatenate_datasets([tokenized_raw_datasets, tokenized_qa_datasets])

# Träningsargument
training_args = TrainingArguments(
    output_dir="./gpt_sw3_finetuned_2",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=10,  
    fp16=True,  
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    learning_rate=5e-5,  
    warmup_steps=500, 
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Konfigurera Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_datasets,
    optimizers=(torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate), 
                get_linear_schedule_with_warmup(torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate),
                                                num_warmup_steps=training_args.warmup_steps,
                                                num_training_steps=len(combined_datasets) // training_args.per_device_train_batch_size))
)

# Starta träningen
trainer.train()

# Spara den finjusterade modellen och tokenizer
model.save_pretrained('./gpt_sw3_finetuned_2')
tokenizer.save_pretrained('./gpt_sw3_finetuned_2')