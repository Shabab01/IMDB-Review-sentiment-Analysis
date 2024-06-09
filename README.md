# IMDB Movie Review Sentiment Analysis Project

## Overview
This project involves building a sentiment analysis model to classify IMDB movie reviews as positive or negative. The project encompasses data preparation, text processing, model building, evaluation, and deployment of a web application using Gradio. The model is built using Long Short Term Memory (LSTM) neural networks.

## Overall Agenda

### 1. Data Preparation and Preprocessing

**Objective**: Prepare the data for analysis and model training.

1. **Loading Data**:
   - Import necessary libraries.
   - Load the IMDB dataset using `pandas`.

2. **Exploratory Data Analysis**:
   - Display the first few records of the dataset.
   - Check the shape and data types.
   - Examine the distribution of sentiment labels.

3. **Data Cleaning and Encoding**:
   - Replace textual sentiment labels (positive/negative) with numerical values (1/0).

### 2. Text Processing

**Objective**: Transform the text data into a format suitable for machine learning models.

1. **Splitting the Dataset**:
   - Split the dataset into training and testing sets.

2. **Tokenization and Padding**:
   - Tokenize the text data using `Tokenizer` from Keras.
   - Convert the text data into sequences of integers.
   - Pad the sequences to ensure uniform length.

### 3. Model Building and Training

**Objective**: Build and train an LSTM model to classify the sentiment of movie reviews.

1. **Building the LSTM Model**:
   - Create a sequential model.
   - Add an embedding layer.
   - Add an LSTM layer with dropout for regularization.
   - Add a dense output layer with a sigmoid activation function.

2. **Compiling the Model**:
   - Compile the model using the Adam optimizer and binary cross-entropy loss function.

3. **Training the Model**:
   - Train the model on the training data.
   - Validate the model using a validation split.

### 4. Model Evaluation

**Objective**: Evaluate the model's performance on the test dataset.

1. **Evaluating the Model**:
   - Evaluate the model's loss and accuracy on the test set.
   - Print the evaluation results.

2. **Saving the Model and Tokenizer**:
   - Save the trained model and tokenizer for future use.

### 5. Building a Predictive System

**Objective**: Create a function to predict the sentiment of new movie reviews using the trained model.

1. **Defining the Predictive Function**:
   - Define a function that takes a movie review as input.
   - Tokenize and pad the review.
   - Predict the sentiment using the trained model.
   - Return the predicted sentiment.

### 6. Developing a Web Application

**Objective**: Develop a web application to allow users to input movie reviews and get sentiment predictions.

1. **Loading the Model and Tokenizer**:
   - Load the saved model and tokenizer.

2. **Creating the Web Interface**:
   - Use the Gradio library to create a user interface for the predictive system.

3. **Launching the Web Application**:
   - Launch the web application and generate a shareable link.

## How to Run the Project

### Prerequisites

- Python 3.x
- pandas
- numpy
- scikit-learn
- TensorFlow/Keras
- joblib
- gradio

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/imdb-sentiment-analysis.git
    cd imdb-sentiment-analysis
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset**:
    Place the `imdb_dataset.csv` file in the project directory.

### Running the Model Training

1. **Run the data preparation and model training script**:
    ```bash
    python train_model.py
    ```

### Running the Web Application

1. **Run the web application script**:
    ```bash
    python app.py
    ```

2. **Access the application**:
    A link will be generated, which you can open in your web browser to interact with the sentiment analysis model.

Structure

```
imdb-sentiment-analysis/
│
├── data/
│   └── imdb_dataset.csv       # Dataset file
│
├── model/
│   ├── model.h5               # Saved LSTM model
│   └── tokenizer.pkl          # Saved tokenizer
│
├── notebooks/
│   └── imdb_sentiment_analysis.ipynb # Jupyter notebook for EDA and model training
│
├── scripts/
│   ├── train_model.py         # Script for data preparation and model training
│   └── app.py                 # Script for running the web application
│
├── README.md                  # Project documentation
├── requirements.txt           # Required packages
└── LICENSE                    # License file
```


## Acknowledgements
- The dataset used in this project is from [Kaggle](https://www.kaggle.com/).
- Thanks to the creators of TensorFlow, Keras, pandas, numpy, and Gradio for their excellent libraries.

## Contact
For any questions or inquiries, please contact [shababahmed69@gmail.com].

---
