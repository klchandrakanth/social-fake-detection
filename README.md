# Social Media Fake Account Detection

This project implements a fake account detection system using deep learning models trained on social media data.

## Project Structure

social-fake-detection/
│-- models/
│-- data/
│-- src/
│   │-- preprocess.py
│   │-- train_model.py
│   │-- evaluate_model.py
│-- train.py
│-- README.md
│-- requirements.txt
│-- .gitignore




## Installation

Clone the repository:
   ```bash
   git clone https://github.com/klchandrakanth/social-fake-detection.git
   cd social-fake-detection


Install dependencies:
   pip install -r requirements.txt


Usage
Train the model:
   python src/train_model.py

Evaluate the model:
   python src/evaluate_model.py


Dataset
The model is trained on a synthetic dataset of social media accounts containing features such as:

Number of posts
Followers count
Profile picture status (Yes/No)
Verification status (Yes/No)