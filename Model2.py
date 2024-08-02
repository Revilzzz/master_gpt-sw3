import torch
import json
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset

# Kontrollera om GPU är tillgänglig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ladda och förbered data
def load_and_prepare_qa_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        qa_data = json.load(file)
    text_data = ["fråga: " + item["question"] + " svar: " + item["answer"] for item in qa_data]
    return Dataset.from_dict({'text': text_data})

# Collate-funktion för att konvertera batcher till tensorer
def collate_fn(batch):
    return {key: torch.tensor([example[key] for example in batch]) for key in batch[0]}

tokenizer = AutoTokenizer.from_pretrained('AI-Sweden-Models/gpt-sw3-356m-instruct')
model = AutoModelForCausalLM.from_pretrained('AI-Sweden-Models/gpt-sw3-356m-instruct')

# Flytta modellen till GPU
model.to(device)

qa_dataset = load_and_prepare_qa_data('output_file_supervised.json')
unsupervised_data = load_dataset('text', data_files={'train': 'C:/gptsw3/gpt-sw3-356m-instruct/txtfiler/*.txt'}, split='train')

# Tokenisera och förbered labels
def tokenize_and_prepare_labels(examples):
    tokenized_inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

qa_dataset = qa_dataset.map(tokenize_and_prepare_labels, batched=True, remove_columns=["text"])
unsupervised_data = unsupervised_data.map(tokenize_and_prepare_labels, batched=True, remove_columns=["text"])

# Skapa DataLoader för supervised och unsupervised data
batch_size = 2
qa_dataloader = DataLoader(qa_dataset, sampler=RandomSampler(qa_dataset), batch_size=batch_size, collate_fn=collate_fn)
unsupervised_dataloader = DataLoader(unsupervised_data, sampler=RandomSampler(unsupervised_data), batch_size=batch_size, collate_fn=collate_fn)

training_args = TrainingArguments(
    output_dir='./results6',
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    evaluation_strategy="epoch", 
)

# Träningsloop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()


print("Starting supervised training...")
for epoch in range(int(training_args.num_train_epochs)):
    print(f"Supervised Epoch {epoch + 1}/{training_args.num_train_epochs}")
    for batch_idx, batch in enumerate(qa_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch_idx % 10 == 0:
            print(f"Supervised QA Batch {batch_idx}, Loss: {loss.item()}")

    
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch_idx, batch in enumerate(qa_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
        avg_loss = total_loss / len(qa_dataloader)
        print(f"Supervised Epoch {epoch + 1} Average Loss: {avg_loss}")
    model.train()


print("Starting unsupervised training...")
for epoch in range(int(training_args.num_train_epochs)):
    print(f"Unsupervised Epoch {epoch + 1}/{training_args.num_train_epochs}")
    for batch_idx, batch in enumerate(unsupervised_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 10 == 0:
            print(f"Unsupervised Batch {batch_idx}, Loss: {loss.item()}")

    
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch_idx, batch in enumerate(unsupervised_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
        avg_loss = total_loss / len(unsupervised_dataloader)
        print(f"Unsupervised Epoch {epoch + 1} Average Loss: {avg_loss}")
    model.train()

# Spara modellen
model.save_pretrained('./results6')
tokenizer.save_pretrained('./results6')
