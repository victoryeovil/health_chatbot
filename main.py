import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gdown
#download the files


# Replace this URL with the shareable link of the file you want to downloadhttps://drive.google.com/file/d/1-U9TxExWFDO5jneQ3vG-oGzTyPzmPjA7/view?usp=sharing
google_drive_url = "https://drive.google.com/file/d/1-13ryvVrNzD8dnGeqUQztLGWMOOeS6_M/view?usp=sharing"

# Specify the output file path
output_file = "./model/pytorch_model.bin"

# Download the file
gdown.download(google_drive_url, output_file, quiet=False)


# Load the fine-tuned model from Google Drive
model = GPT2LMHeadModel.from_pretrained("./model/")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

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

# Streamlit app
st.title("GPT-2 Chatbot")

user_input = st.text_input("You:", value="")
if st.button("Generate Response"):
    if user_input:
        response = generate_response(user_input, max_length=50)
        st.write("Chatbot:", response)

st.markdown("Enter 'exit' to end the chat.")
