# 🤖 Machine Learning Project: Implement a Neural Network for Sentiment Analysis

In this project, you will build and train a **feedforward neural network** to perform **sentiment classification** on text data. Instead of sequence models, you will use TF-IDF to convert text into numerical feature vectors, then train a traditional neural network on these vectors. You will evaluate your model's performance on training and test sets to assess overfitting and improve generalization.

---

### 📌 Project Tasks

* Load the book review dataset
* Define the sentiment label and identify features
* Create labeled examples from the dataset
* Split data into training and test sets
* Transform text data using TF-IDF vectorization
* Construct a feedforward neural network model using Keras
* Train the neural network on the training data
* Compare training and validation performance to check for overfitting
* Improve model generalization via tuning and techniques
* Evaluate final model performance on the test set

---

### 🛠️ Example Code Snippet

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset
reviews_df = pd.read_csv('book_reviews.csv')

# Define label and features
label = 'sentiment'  # e.g., positive/negative label column
texts = reviews_df['review_text']

# Create labeled examples
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, reviews_df[label], test_size=0.2, random_state=42
)

# Transform text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(X_train_texts).toarray()
X_test = vectorizer.transform(X_test_texts).toarray()

# Build neural network model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')
```
