from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load the tokenizer for your GPT model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define your text here
text = """Sexually transmitted infections (STIs) are spread predominantly by unprotected sexual contact. Some STIs can also be transmitted during pregnancy, childbirth and breastfeeding and through infected blood or blood products.

STIs have a profound impact on health. If untreated, they can lead to serious consequences including neurological and cardiovascular disease, infertility, ectopic pregnancy, stillbirths, and increased risk of Human Immunodeficiency Virus (HIV). They are also associated with stigma, domestic violence, and affects quality of life.

The majority of STIs have no symptoms. When they are present common symptoms of STIs are vaginal or urethral discharge, genital ulcer and lower abdominal pain.

The most common and curable STIs are trichomonas, chlamydia, gonorrhoea and syphilis. Rapidly increasing antimicrobial resistance is a growing threat for untreatable gonorrhoea.

Viral STIs including HIV, genital herpes simplex virus (HSV), viral hepatitis B, human papillomavirus (HPV) and human T-lymphotropic virus type 1 (HTLV-1) lack or have limited treatment options. Vaccines are available for hepatitis B to prevent infection that can lead to liver cancer and for HPV to prevent cervical cancer. HIV, HSV and HTLV-1 are lifelong infections: for HIV and HSV there are treatments that can suppress the virus, but currently there are no cures for any of these viral STIs.

Condoms used correctly and consistently are effective methods to protect against STIs and HIV. Screening with early diagnosis of people with STIs and their sexual partners offers the best opportunity for effective treatment and for preventing complications and further transmission."""

# Tokenize the text
tokenized_text = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=1024)

# Convert tokenized_text to PyTorch tensors
input_ids = torch.tensor([tokenized_text])

# Create a DataLoader
dataset = TensorDataset(input_ids)
train_dataloader = DataLoader(dataset, batch_size=1)

# Load a pre-trained GPT model
model_name = "gpt2" 
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader))

# Fine-tuning loop
num_epochs = 3  # You can adjust this
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        input_ids = batch[0].to(device)
        labels = batch[0].to(device)

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model_dir = "./model/"
model.save_pretrained(model_dir)

# Load the fine-tuned model
model = GPT2LMHeadModel.from_pretrained(model_dir)

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
