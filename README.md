# Text-Classification-using-Hugging-Face
Text classification is a common NLP task that assigns a label or class to text. Some of the largest companies run text classification in production for a wide range of practical applications.

Objective: The goal is to build a text classification model using the Hugging Face library to classify a dataset of text into one of multiple categories. The candidate will use a pre-trained model such as BERT or GPT-2 as a starting point and fine-tune it on the classification task.

The chosen dataset has multiple categories (e.g. news articles labeled as sports, politics, entertainment, etc.). The dataset should have at least 1000 samples for each category. Preprocess the text data by cleaning it, removing stopwords, punctuations and other irrelevant characters. Use the Hugging Face library to fine-tune a pre-trained model such as BERT or GPT-2 on the classification task. The candidate should use the transformers library in python. Train the model on the dataset and evaluate the performance using metrics such as accuracy, precision, recall and F1-score.

The task is to classify a dataset of Yelp reviews into one of five categories, namely, 1-star, 2-star, 3-star, 4-star, and 5-star ratings. The dataset used for this task is the Yelp reviews dataset, which consists of over 6 million reviews from businesses across 11 metropolitan areas in four countries.

The preprocessing steps taken:

    • The text was converted to lowercase.
    • Punctuation and special characters were removed.
    • Stop words were removed.
    • The text was tokenized using the WordPiece tokenizer from the BERT model.
    • The tokens were converted to their corresponding IDs using the BERT tokenizer.
    • The input sequences were padded or truncated to a fixed length of 256.
    
Model Architecture and Fine Tuning:
The pre-trained model used for this task is the BERT model. The BERT model was fine-tuned for the Yelp reviews classification task using the Hugging Face library. The architecture of the model consists of the BERT base model followed by a classifier layer. The classifier layer consists of a linear layer followed by a softmax activation function. The model was trained using the bert-base-uncased optimizer with a learning rate of 2e-5 and a batch size of 32. The model was trained for 3 epochs.

Evaluation Metrics and Results:
The evaluation metrics used for this task is accuracy, precision, recall, F1-score. The model achieved an accuracy of 73.8%, precision of 75.9%, recall score of 73.5%, F1-score of 75.1% on the test set.

The model achieved a moderate accuracy of 73.8% on the test set. However, there is still room for improvement. One possible way to improve the performance of the model is to use a larger pre-trained model such as BERT-large or GPT-2. Another way to improve the performance of the model is to use data augmentation techniques such as back-translation or word replacement.
