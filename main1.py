from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load the tokenizer for your GPT model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define your text here
text = "Your text here"

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
