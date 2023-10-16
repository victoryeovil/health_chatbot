import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gdown
import os
import requests

# Replace 'YOUR_FILE_URL' with the direct download link you obtained from Google Drive.
file_url = "https://drive.google.com/drive/folders/1LcyoTshbw7QCXKbRVbTku0VULTaDi-OK?usp=sharing"
output_path = "model"  # Specify the path where you want to save the downloaded model files.

gdown.download(file_url, output_path, quiet=False)

r = requests.get(file_url, stream=True)
with open('./pytorch_model.bin', 'wb') as f:
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)
print("Finished downloading model.")


# Load the fine-tuned model
link = "https://drive.google.com/drive/folders/1LcyoTshbw7QCXKbRVbTku0VULTaDi-OK?usp=sharing"
model = GPT2LMHeadModel.from_pretrained("/model/")
tokenizer = GPT2Tokenizer.from_pretrained("/model/")

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate a response
def generate_response(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Main loop for interacting with the chatbot
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = generate_response(user_input, max_length=50)
    print("Chatbot:", response)
