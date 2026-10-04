# Consumer Complaint Timeliness Predictor

## What it does
Predicts whether a consumer complaint will receive a timely response,
using the CFPB Consumer Complaints dataset (18M+ rows).

## Why it matters
High accuracy can hide a model that fails at the task that matters.
This project demonstrates that trap and how to detect it with a
confusion matrix.

## How to run it
1. Place complaints.csv in the same folder (Link below)
2. python Consumer_Complaint_Code.py

## Results
- 99.51% accuracy — but only 25% recall on the rare class
- Precision of ~80% on flagged late responses
- A clear example of why accuracy alone misleads


Link to sample data used: https://catalog.data.gov/dataset/consumer-complaint-database


"""

