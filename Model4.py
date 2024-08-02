import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, DatasetDict

# Kontrollera om GPU är tillgänglig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ladda och förbered QA data
def load_and_prepare_qa_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        qa_data = json.load(file)
    text_data = [{"text": f"Context: {item['context']}\nFråga: {item['question']}\nSvar: {item['answer']}"} for item in qa_data]
    return Dataset.from_dict({'text': [item['text'] for item in text_data]})

tokenizer = AutoTokenizer.from_pretrained('AI-Sweden-Models/gpt-sw3-356m-instruct')
model = AutoModelForCausalLM.from_pretrained('AI-Sweden-Models/gpt-sw3-356m-instruct')

# Flytta modellen till GPU
model.to(device)

# Ladda dataset
qa_dataset = load_and_prepare_qa_data('output_file_supervised.json')
dataset_split = qa_dataset.train_test_split(test_size=0.1)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']

# Tokenisera och förbered labels för träningsdata
def tokenize_and_prepare_labels(examples):
    texts = examples['text']
    tokenized_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

train_dataset = train_dataset.map(tokenize_and_prepare_labels, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_and_prepare_labels, batched=True, remove_columns=["text"])


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Träningsparametrar
training_args = TrainingArguments(
    output_dir='./results15',
    num_train_epochs=30,
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=2,  
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine"
)

# Träningsloop för enbart superviserad data
print("Starting supervised training...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

# Spara modellen
model.save_pretrained('./results15')
tokenizer.save_pretrained('./results15')

# Generera text
def generate_text(prompt, max_length=100):
    encoded_input = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output_sequences = model.generate(
        input_ids=encoded_input,
        max_length=max_length,
        temperature=0.7,
        do_sample=True,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text

print("Finetunead version of GPT-SW3 modellen. Allt information som modellen matar ut bör kontrolleras.")
while True:
    user_input = input("Du: ")
    if user_input.lower() == 'exit':
        break
    
    prompt = f"Fråga: {user_input}\nSvar:"
    response = generate_text(prompt)
    print("GPT-SW3:", response)
