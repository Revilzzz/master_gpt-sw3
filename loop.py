from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import os
import json

#Ange sökväg till modellen
model_checkpoint_path = './Model4'
generation_config_path = 'generation_config.json'


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path)

# Ladda generation config manuellt
with open(generation_config_path, 'r') as f:
    generation_config_data = json.load(f)

def generate_text(prompt):
    # Kodera prompten till token-IDs
    encoded_input = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generera output från modellen
    output_sequences = model.generate(
        input_ids=encoded_input,
        **generation_config_data  
    )
    
    # Dekodera texten tillbaka till strängar
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text

print("Finetunead version of GPT-SW3 modellen. Allt information som modellen matar ut bör kontrolleras.")
while True:
    user_input = input("Du: ")
    if user_input.lower() == 'exit':
        break
    

    prompt = f"fråga: {user_input} svar:"
    response = generate_text(prompt)
    print("GPT-SW3:", response)
