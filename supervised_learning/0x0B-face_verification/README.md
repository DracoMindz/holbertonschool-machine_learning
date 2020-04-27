#0x0B Face Verification
Specializations - Machine Learning ― Supervised Learning

##Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:
```
General
What is face recognition?
What is face verification?
What is dlib?
How to use dlib for detecting facial landmarks
What is an affine transformation?
What is face alignment?
Why would you use facial alignment?
How to use opencv-python for face alignment
What is one-shot learning?
What is triplet loss?
How to create custom Keras layers
How to add losses to a custom Keras layer
How to create a Face Verification system
What are the ethical quandaries regarding facial recognition/verification?
```


For this project dlib 19.17.0 was a required install.

##Tasks

***0. Load  Images***

Write a function that loads images fom a directory or file.

***1. Load CSV***

Write a function that loads the contents of a csv file as a list of lists.

***2. Initialize Face Align***

Create the class FaceAlign

***3. Detect Faces***

Update the class FaceAlign - Method detects aface in an image

***4. Find Landmarks***

Update the class FaceAlign - Method finds facial landmarks

***5. Align Faces***

Update the class FaceAlign - Method aligns an image for face verification

***6. Save Files***

Write a function that saves images to a specific path

***7. Gradient Triplets***

Write a function that generates triplets

***8. Initialize Triplet Loss***

Create a custom layer class the inherits from tensorflow.keras.layers.Layer

***9. Calculate Triplet Loss***

Update the class TripletLoss- Method returns a tensor containing the triplet loss values

***10. Call Triplet Loss***

Update the class TripletLoss - Method returns the triplet loss tensor

***11. Initialize Train Model***

Create a Class: Method trains a model for face verification using triplet loss

***12. Train***

Update the class TrainModel - returns the History output from the training

***13. Save***

Update the class TrainModel - Method saves the base embedding model

***14. Calculate Metrics***

Update the class TrainModel - Static Methods return the f1 score, the accuracy

***15. Best Tau***

Update the class TrainModel - returns (tau, f1, acc)

***16. Initialize Face Verification***

Create the class FaceVerification 

***17. Embedding***

Update the class FaceVerififcation - Returns a numpy.ndarray of embeddings 

***18. Verify***

Update the class FaceVerification - with public instance method




