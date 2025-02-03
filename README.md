This project focuses on building a machine learning model to classify SMS messages as spam or legitimate using natural language processing (NLP) techniques. The dataset used for this task is sourced from Kaggle, provided by CodeSoft.

The model uses techniques like TF-IDF (Term Frequency-Inverse Document Frequency) or word embeddings for feature extraction, along with classifiers such as Naive Bayes, Logistic Regression, or Support Vector Machines (SVM) to perform the classification.

Table of Contents
	•	Project Overview
	•	Dataset
	•	Modeling
	•	Techniques Used
	•	Requirements
	•	Installation
	•	Usage
	•	Evaluation
	

Project Overview

Spam SMS detection is a key task in ensuring mobile security and preventing unwanted messages from reaching users. This project aims to develop a machine learning model that can classify SMS messages as spam or legitimate based on their content.

The dataset consists of labeled SMS messages that are preprocessed and used to train different machine learning models. The goal is to identify spam messages and block them from the user’s inbox.

Dataset

The dataset used in this project is sourced from Kaggle and provided by CodeSoft. It contains the following attributes:
	•	Message: The SMS message content.
	•	Label: The target variable, where 1 represents spam and 0 represents a legitimate (non-spam) message.

Dataset link: (You can provide the specific Kaggle dataset link here if available)

Modeling

For the SMS spam detection task, the following machine learning algorithms were applied:
	1.	Naive Bayes: A probabilistic classifier that is often used for text classification tasks.
	2.	Logistic Regression: A linear model for binary classification, which works well with text data.
	3.	Support Vector Machines (SVM): A robust classifier that works by finding the optimal hyperplane that separates spam and legitimate messages.

To extract features from the text messages, two different techniques were explored:
	•	TF-IDF (Term Frequency-Inverse Document Frequency): A statistical method used to evaluate the importance of a word in a document relative to a corpus of documents.
	•	Word Embeddings: Pre-trained word vectors (like Word2Vec or GloVe) are used to convert words into numerical representations that capture semantic meaning.

Techniques Used

1. TF-IDF (Term Frequency-Inverse Document Frequency)
	•	TF-IDF is used to transform the raw text messages into numeric features. The technique calculates how important a word is in the entire corpus based on its frequency in the document and its rarity across all documents.

2. Word Embeddings
	•	Word embeddings like Word2Vec or GloVe convert words into vector representations, capturing the semantic meaning of words. These embeddings can be used as input to machine learning classifiers.

3. Classifiers
	•	Naive Bayes: A probabilistic classifier commonly used for text classification, works well for small datasets and performs efficiently.
	•	Logistic Regression: A simple, yet effective linear classifier.
	•	Support Vector Machines (SVM): A classifier that works by finding the best boundary (hyperplane) that divides the spam and legitimate messages.

Requirements

To run this project, you’ll need the following Python libraries:
	•	Python 3.x
	•	pandas
	•	numpy
	•	scikit-learn
	•	nltk
	•	matplotlib
	•	seaborn
	•	tensorflow (if using word embeddings)

You can install the required libraries using the following:

pip install pandas numpy scikit-learn nltk matplotlib seaborn tensorflow

Installation
	1.	Clone the repository:

git clone https://github.com/your-username/spam-sms-detection.git
cd spam-sms-detection


	2.	Install the required dependencies:

pip install -r requirements.txt


	3.	Download the dataset from Kaggle and place it in the data/ directory.

Usage
	1.	Load and Preprocess the Data:
	•	The data is cleaned, including tokenization, removing stop words, and other preprocessing steps to prepare the text data for modeling.
	2.	Feature Extraction:
	•	Use TF-IDF to convert the text data into a numerical format.
	•	Alternatively, use Word Embeddings (Word2Vec or GloVe) to transform the text into dense vector representations.
	3.	Train the Model:
	•	The models (Naive Bayes, Logistic Regression, SVM) are trained on the processed dataset.
	4.	Evaluate the Model:
	•	The models are evaluated based on their accuracy, precision, recall, and F1-score.

To run the script and train the model:

python train_model.py

Evaluation

The performance of the models is evaluated using the following metrics:
	•	Accuracy: Percentage of correctly classified messages.
	•	Precision: Proportion of true positives to all predicted positives.
	•	Recall: Proportion of true positives to all actual positives.
	•	F1-Score: The harmonic mean of precision and recall, providing a balanced evaluation.

