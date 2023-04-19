### Facial Emotion Tracking Capstone Project

# Introduction
This capstone project is about building a facial emotion tracking model that can predict human emotions from facial expressions. The model is trained on a dataset containing images of faces with labeled emotions. The goal is to help stakeholders like therapists, educators, and researchers better understand people's emotions in various contexts. This project includes a Jupyter notebook that walks through the process of data preparation, model building, and evaluation, as well as an app built with Streamlit that uses the trained model to make predictions on user-uploaded images.

# Data Understanding
The dataset used in this project contains grayscale images of faces with labeled emotions. The data is sourced from the "icml_face_data.csv" file, which is suitable for this project due to the diversity of emotions and facial expressions. The notebook presents the size of the dataset and descriptive statistics for all features used in the analysis, justifies the inclusion of features based on their properties and relevance, and identifies any limitations of the data that have implications for the project.

# Data Preparation
The notebook.ipynb file shows how the data is prepared for analysis. It includes instructions and code for loading and preprocessing the raw data, as well as comments and text to explain each step. The steps taken in data preparation are appropriate for the problem being solved.

# Modeling
The notebook.ipynb file demonstrates an iterative approach to model-building. It starts with a simple baseline model and progressively introduces new models that improve upon prior models. The results of each model are interpreted, and changes are explicitly justified based on the results of prior models and the problem context. Improvements found from running new models are also described.

# Evaluation
The notebook.ipynb file shows how well the final model solves the real-world problem. The choice of evaluation metrics is justified using the context of the real-world problem and the consequences of errors. One final model is identified based on its performance on the chosen metrics with validation data. The performance of the final model is evaluated using holdout test data.

The results for the final model (v2) are as follows:

.Test loss (v2): 1.0121
.Test accuracy (v2): 0.6478
The confusion matrix for the final model (v2) is shown below: