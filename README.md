Machine Learning & AI Assignment

This project demonstrates applications of Classical Machine Learning, Deep Learning, and Natural Language Processing (NLP) using popular Python frameworks — Scikit-learn, TensorFlow/PyTorch, and spaCy. It also includes an Ethics & Optimization section focusing on fairness and debugging.

📘 Task 1: Classical ML with Scikit-learn
Dataset: Iris Species Dataset
Goal:

Train a Decision Tree Classifier to predict the species of an iris flower based on its measurements.

Steps Performed:

Data Loading:

Imported the Iris dataset using sklearn.datasets.load_iris().

Preprocessing:

Checked for and handled missing values (none found).

Encoded species labels using LabelEncoder.

Split the dataset into training (80%) and testing (20%) sets.

Model Training:

Trained a DecisionTreeClassifier using the training data.

Tuned hyperparameters like criterion, max_depth, and random_state.

Model Evaluation:

Evaluated performance using:

Accuracy

Precision

Recall

Generated a classification report and confusion matrix.

Results:

Achieved high accuracy (>90%) on test data.

Precision and recall values were strong across all species classes.

Deliverable:

iris_decision_tree.ipynb or iris_decision_tree.py – includes commented code explaining each step.

🧩 Task 2: Deep Learning with TensorFlow / PyTorch
Dataset: MNIST Handwritten Digits
Goal:

Build a Convolutional Neural Network (CNN) to classify digits (0–9) with over 95% accuracy.

Steps Performed:

Data Preparation:

Loaded MNIST using tf.keras.datasets.mnist or torchvision.datasets.MNIST.

Normalized pixel values to range [0, 1].

Reshaped images to include a channel dimension (28×28×1).

Model Architecture:

Convolutional Layers: Extract features.

Pooling Layers: Reduce spatial dimensions.

Dense Layers: Perform classification.

Activation Functions: ReLU and Softmax.

Training:

Compiled with Adam optimizer and sparse categorical crossentropy loss.

Trained for several epochs until test accuracy exceeded 95%.

Evaluation & Visualization:

Evaluated model on test data.

Displayed predictions on 5 sample images with true vs predicted labels.

Results:

Final Test Accuracy: ≈98%

Model correctly predicted most digits with minimal confusion.

Deliverable:

mnist_cnn_tf.ipynb or mnist_cnn_torch.py – includes model architecture, training loop, and evaluation.

💬 Task 3: NLP with spaCy
Dataset: Amazon Product Reviews
Goal:

Perform Named Entity Recognition (NER) to extract product names and brands, then analyze sentiment using a rule-based approach.

Steps Performed:

Text Loading:

Collected or simulated user reviews from Amazon.

NER Processing:

Used en_core_web_sm model from spaCy.

Extracted named entities labeled as PRODUCT, ORG, etc.

Sentiment Analysis:

Implemented a rule-based sentiment system:

Counted positive/negative words using a simple lexicon.

Classified review sentiment as Positive, Negative, or Neutral.



Deliverable:

amazon_reviews_ner.ipynb or amazon_reviews_ner.py – includes entity extraction and sentiment results.

⚖️ Part 3: Ethics & Optimization
1. Ethical Considerations

Potential Biases:

MNIST Bias: The dataset consists mostly of clean, centered digits. Models trained on it may fail with messy handwriting or digits from other writing systems.

Amazon Reviews Bias: Reviews may be biased toward popular products or languages, leading to skewed sentiment results.

Mitigation Tools:

TensorFlow Fairness Indicators:

Evaluate model fairness across demographic slices or data subgroups.

spaCy’s Rule-based Systems:

Customize patterns to avoid cultural or linguistic misinterpretations of sentiment.

2. Troubleshooting Challenge

Identified and fixed TensorFlow errors like:

Dimension mismatches (reshaping inputs correctly).

Incorrect loss functions (used sparse_categorical_crossentropy for integer labels).

Optimizer misconfiguration (ensured correct learning rate and compilation).

Validated by rerunning the training loop successfully and confirming improved accuracy.

Requirements

Install dependencies before running:

pip install numpy pandas scikit-learn tensorflow torch torchvision spacy matplotlib
python -m spacy download en_core_web_sm

Folder Structure
project/
│
├── Task1_Iris_DecisionTree/
│   ├── iris_decision_tree.ipynb
│   └── README.md
│
├── Task2_MNIST_CNN/
│   ├── mnist_cnn_tf.ipynb
│   └── README.md
│
├── Task3_NLP_spaCy/
│   ├── amazon_reviews_ner.ipynb
│   └── README.md
│
└── Ethics_Optimization/
    ├── ethics_report.md
    └── debugged_tensorflow_script.py

 Author

Name: Benard Odudo
AI Tools and Frameworks
Date: October 2025