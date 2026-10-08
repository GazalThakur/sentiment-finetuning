from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


# ============================================================
# 1. Choose device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. Load the IMDb dataset
# ============================================================

dataset = load_dataset("stanfordnlp/imdb")

print("\nDataset:")
print(dataset)


# ============================================================
# 3. Load DistilBERT tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)


# ============================================================
# 4. Tokenization function
# ============================================================

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True
    )


# ============================================================
# 5. Tokenize the dataset
# ============================================================

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)

print("\nTokenized dataset:")
print(tokenized_dataset)


# ============================================================
# 6. Create data collator
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# 7. Create DataLoaders
# ============================================================

train_dataloader = DataLoader(
    tokenized_dataset["train"],
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator
)

test_dataloader = DataLoader(
    tokenized_dataset["test"],
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator
)


# ============================================================
# 8. Load DistilBERT
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

model.to(device)

print("\nModel loaded!")
print("Model device:", next(model.parameters()).device)


# ============================================================
# 9. Create optimizer
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=5e-5
)


# ============================================================
# 10. Training
# ============================================================

epochs = 2

print("\nStarting training...\n")

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for step, batch in enumerate(train_dataloader):

        # Move batch to GPU
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        # Forward pass
        outputs = model(**batch)

        # Get loss
        loss = outputs.loss

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Clear old gradients
        optimizer.zero_grad()

        # Track loss
        total_loss += loss.item()

        # Print progress
        if (step + 1) % 500 == 0:
            print(
                f"Epoch {epoch + 1} | "
                f"Step {step + 1} | "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_dataloader)

    print(
        f"\nEpoch {epoch + 1} finished!"
    )

    print(
        f"Average training loss: {average_loss:.4f}\n"
    )


# ============================================================
# 11. Evaluation
# ============================================================

print("Evaluating model...\n")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch in test_dataloader:

        # Move batch to GPU
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        # Forward pass
        outputs = model(**batch)

        # Get predictions
        predictions = torch.argmax(
            outputs.logits,
            dim=-1
        )

        # Compare predictions with labels
        correct += (
            (predictions == batch["labels"])
            .sum()
            .item()
        )

        total += batch["labels"].size(0)


accuracy = correct / total

print(f"Test accuracy: {accuracy:.4f}")
print(f"Test accuracy: {accuracy * 100:.2f}%")


# ============================================================
# 12. Save the fine-tuned model
# ============================================================

save_directory = "./sentiment-distilbert"

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print("\nModel saved to:", save_directory)