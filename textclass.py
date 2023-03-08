import pandas as pd
import numpy as np
from transformers import AutoTokenizer

# Load the dataset
df = pd.read_csv("yelp.csv")
# Preprocess the data
# Convert the data to a format suitable for training
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(list(df['text']), truncation=True, padding=True)
labels = df['stars'].values

from transformers import AutoModelForSequenceClassification

# Load the pre-trained model and modify it for classification
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(labels)))

from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

# Split the dataset into training and validation sets
train_encodings, val_encodings, train_labels, val_labels = train_test_split(encodings, labels, test_size=1/5, random_state=0)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=(train_encodings, train_labels),
    eval_dataset=(val_encodings, val_labels)
)

trainer.train()

# Save the best model
trainer.save_model('./saved_model')

from transformers import pipeline, AutoTokenizer

# Load the saved model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('./saved_model')
model = AutoModelForSequenceClassification.from_pretrained('./saved_model')

# Define the text classification pipeline
text_classification = pipeline(
    'text-classification',
    model=model,
    tokenizer=tokenizer
)

# Use the model for predictions
input_text = "I had an amazing experience at this restaurant. The food was delicious, and the service was excellent. I would definitely give it a 5-star rating."
predicted_category = text_classification(input_text)

#Evaluating test metrics

ac = accuracy_score(eval_dataset, input_text)
pr = precision_score(eval_dataset, input_text)
re = recall_score(eval_dataset, input_text)
f1 = f1_score(eval_dataset, input_text)

print("Accuracy : ",ac)
print("Precision : ",pr)
print("Recall : ",re)
print("F1 Score : ",f1)
