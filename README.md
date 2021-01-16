# Twitter Sentiment Analysis with Sentiment140 Dataset

## Introduction
**Sentiment Analysis** is one of the key applications of **Natural Language Processing (NLP)**, which is an interdisciplinary field that sits at the intersection of artificial intelligence and linguistics. NLP is primarily concerned with the automatic processing and understanding of natural language by computers, and a sentiment analysis model is a model that is able to predict if a given piece of text expresses a positive or negative sentiment.

In this notebook, we will attempt to train a sentiment analysis model using the Sentiment140 dataset, which is a collection of 1.6 million annotated tweets.

## Business Problem

As more and more consumers shift to digital and spend more time online, it has become increasingly important for brands to engage with their customers online using social media platforms such as Facebook, Instagram and Twitter. These public platforms contain lots of customer reviews regarding the brands' products or services, and they tend to have a huge impact on potential customers' purchasing decisions. Negative reviews on a brand's social media presence will no doubt have a significant damaging impact on the company's sales and reputation.

Hence, it is imperative for companies to accurately identify the public sentiment towards their brands in real time. This will then allow them to come up with timely strategic changes and key improvements to their products or services which will help in maximising customer satisfaction.

## Executive Summary

From the results above, we arrive at the following key points regarding the sentiment analysis model:

- We can develop a decent sentiment analysis model using either **TF-IDF** or **word embeddings** as features.
- The average **accuracy** of the top two models (logistic regression and LSTM network) is **77%** on the test set, which means that out of every four tweets, our models are able to accurately predict the sentiments of three of them.
- The average **false positive rate** of the top two models is **23%** on the test set, which means that out of every four negative tweets, our models will incorrectly predict one of them as positive.

The performance of the models can potentially be improved via the following methods:

- **Sample size**: Train on the entire collection of 1.6 million tweets instead of just using a small subset of 200,000 tweets due to limited computing power.
- **TF-IDF**: Experiment with different hyperparameter values for 'ngram_range' and 'max_features' in TfidfVectorizer.
- **Word embedding**: Try out other word embeddings such as Word2vec or other versions of GloVe.
- **Hyperparameter tuning**: Conduct hyperparameter tuning for logistic regression, support vector classifier, Naive Bayes classifier and LSTM network.
- **Model architecture**: Experiment with different model architectures for LSTM network.

With a sentiment analysis model, companies will be able to accurately identify the public sentiment towards their brands in real time. This will then allow them to come up with timely strategic changes and key improvements to their products or services which will help in maximising customer satisfaction.
