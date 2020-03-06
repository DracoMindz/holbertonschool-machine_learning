# 0x04. Error Analysis
Specializations - Machine Learning ― Supervised Learning

## Learning Objectives

### General
```
What is the confusion matrix?
What is type I error? type II?
What is sensitivity? specificity? precision? recall?
What is an F1 score?
What is bias? variance?
What is irreducible error?
What is Bayes error?
How can you approximate Bayes error?
How to calculate bias and variance?
How to create a confusion matrix?
```

### Tasks

***0. Create Confusion mandatory***

Write the function def create_confusion_matrix(labels, logits): that creates a confusion matrix:

***1. Sensitivity mandatory***

Write the function def sensitivity(confusion): that calculates the sensitivity for each class in a confusion matrix

***2. Precision mandatory***

Write the function def precision(confusion): that calculates the precision for each class in a confusion matrix

***3. Specificity mandatory***

Write the function def specificity(confusion): that calculates the specificity for each class in a confusion matrix

***4. F1 score mandatory***

Write the function def f1_score(confusion): that calculates the F1 score of a confusion matrix

***5. Dealing with Error mandatory***

In the text file 5-error_handling, write the lettered answer to the question of how you should approach the following scenarios. Please write the answer to each scenario on a different line. If there is more than one way to approach a scenario, please use CSV formatting and place your answers in alphabetical order (ex. A,B,C)

***6. Compare and Contrast mandatory***

Given the following training and validation confusion matrices and the fact that human level performance has an error of ~14%, determine what the most important issue is and write the lettered answer in the file
