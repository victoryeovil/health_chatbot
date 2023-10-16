text = """Sexually transmitted infections (STIs) are spread predominantly by unprotected sexual contact. Some STIs can also be transmitted during pregnancy, childbirth and breastfeeding and through infected blood or blood products.

STIs have a profound impact on health. If untreated, they can lead to serious consequences including neurological and cardiovascular disease, infertility, ectopic pregnancy, stillbirths, and increased risk of Human Immunodeficiency Virus (HIV). They are also associated with stigma, domestic violence, and affects quality of life.

The majority of STIs have no symptoms. When they are present common symptoms of STIs are vaginal or urethral discharge, genital ulcer and lower abdominal pain.

The most common and curable STIs are trichomonas, chlamydia, gonorrhoea and syphilis. Rapidly increasing antimicrobial resistance is a growing threat for untreatable gonorrhoea.

Viral STIs including HIV, genital herpes simplex virus (HSV), viral hepatitis B, human papillomavirus (HPV) and human T-lymphotropic virus type 1 (HTLV-1) lack or have limited treatment options. Vaccines are available for hepatitis B to prevent infection that can lead to liver cancer and for HPV to prevent cervical cancer. HIV, HSV and HTLV-1 are lifelong infections: for HIV and HSV there are treatments that can suppress the virus, but currently there are no cures for any of these viral STIs.

Condoms used correctly and consistently are effective methods to protect against STIs and HIV. Screening with early diagnosis of people with STIs and their sexual partners offers the best opportunity for effective treatment and for preventing complications and further transmission."""

# Import necessary libraries
from transformers import GPT2Tokenizer

# Load the tokenizer for your GPT model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Read and preprocess your text data
#with open("Health.txt", "r", encoding="utf-8") as file:
#    text_data = file.read()

# Tokenize the text
tokenized_text = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=1024)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"  # You can use a different model, e.g., "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define data loaders, batch size, and other training parameters
# This step can be highly customized depending on your specific use case and hardware resources.

# Example of creating a PyTorch DataLoader (you may need to install torch)
import torch
from torch.utils.data import DataLoader, TensorDataset

# Convert tokenized_text to PyTorch tensors
input_ids = torch.tensor([tokenized_text])

# Create a DataLoader
dataset = TensorDataset(input_ids)
train_dataloader = DataLoader(dataset, batch_size=1)  # Batch size of 1 for simplicity

# Define any additional training parameters, such as the number of epochs, learning rate, etc.
