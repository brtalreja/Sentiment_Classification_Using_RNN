# Sentiment Classification using RNN on SST dataset

## Introduction

This project focuses on sentiment classification using Recurrent Neural Networks (RNNs) on the Stanford Sentiment Treebank (SST) dataset. The goal is to classify movie reviews as positive or negative based on their textual content. Sentiment analysis is a crucial task in Natural Language Processing (NLP) with applications in various domains like customer feedback analysis, social media monitoring, and more.

**Research Questions:**
1. How effective are RNNs in classifying sentiments in movie reviews?
2. Can the performance of the RNN model be improved using various optimization techniques and architectures?
3. What are the trade-offs between different model configurations and hyperparameters in terms of accuracy and overfitting?

## Background and Related Work

Sentiment analysis has been a popular research topic in NLP for many years. Early approaches relied on bag-of-words models and traditional machine learning algorithms such as Support Vector Machines (SVMs) and Naive Bayes. However, with the advent of deep learning, models like RNNs and their variants (LSTM, GRU) have shown significant improvements in capturing the sequential nature of text data.

The Stanford Sentiment Treebank (SST) dataset is a widely used benchmark for sentiment analysis. It contains phrases and sentences from movie reviews annotated with sentiment labels. Previous work has demonstrated that deep learning models, particularly those incorporating RNNs, can achieve state-of-the-art results on this dataset.

## Methods

The project involves the following steps:

1. **Data Preparation:**
   - Load and preprocess the SST dataset using `torchtext`.
   - Create vocabulary dictionaries for text and labels.

2. **Model Definition:**
   - Define an RNN-based model (`RNNClassifier`) for sentiment classification.
   - Experiment with different architectures and hyperparameters (e.g., LSTM, bidirectional RNNs, dropout).

3. **Training and Evaluation:**
   - Train the model using the training data and evaluate on the validation set.
   - Implement various optimization techniques to address overfitting and improve performance.
   - Save the best model based on validation performance.

4. **Testing:**
   - Evaluate the final model on the test set to determine its generalization performance.

## Results and Conclusion

The results of the model training and evaluation are summarized below:

- Initial Model:
  - Train Accuracy: 87.52%
  - Validation Accuracy: 53.50%
  - Test Accuracy: 53.94%

- Optimized Model with Dropout and Adam Optimizer:
  - Train Accuracy: 98.67%
  - Validation Accuracy: 57.95%
  - Test Accuracy: 58.88%

- Further Optimized Model with Bidirectional RNN and Adam Optimizer:
  - Train Accuracy: 99.13%
  - Validation Accuracy: 63.24%
  - Test Accuracy: 64.76%

The experiments show that incorporating techniques like dropout and bidirectional RNNs significantly improves the model's performance by reducing overfitting and capturing more contextual information from the text.

## References

### Data Set:
- Stanford Sentiment Treebank (SST): [SST Dataset](https://nlp.stanford.edu/sentiment/index.html)

### Programming Resource:
- PyTorch: [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- TorchText: [TorchText Documentation](https://pytorch.org/text/stable/index.html)

### Literature Review:
- Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., & Potts, C. (2013). Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP).
- Goldberg, Y. (2016). A Primer on Neural Network Models for Natural Language Processing. Journal of Artificial Intelligence Research, 57, 345-420.
