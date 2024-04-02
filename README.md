# FakeNews-Detection

This project aims to build a machine learning model for detecting fake news articles. The model is trained on a dataset of news articles labeled as either "true" (genuine news) or "false" (fake news).

## Dataset

The dataset used in this project is the `HAI817_Projet_train.csv` file, which contains news articles along with their corresponding labels. The dataset is initially imbalanced, with the following class distribution:

- `false`: 578 instances
- `true`: 211 instances
- `mixture`: 358 instances (a mix of true and false information)
- `other`: 117 instances (unclassified)

## Data Preprocessing

To address the class imbalance issue, the dataset is downsampled by removing instances from the majority class (`false`). Upsampling (duplicating instances from the minority class) was avoided to prevent overfitting.

The following preprocessing steps are applied to the text data:

1. **Language Filtering**: Remove texts that are not in English.
2. **Word Cloud Generation**: Generate word clouds for visualization purposes.
3. **Tokenization**: Convert texts into tokens.
4. **Part-of-Speech Tagging**: Assign part-of-speech tags to words.
5. **Filtering by Part-of-Speech**: Filter words based on their part-of-speech tags (e.g., keep only nouns, verbs, and adjectives).
6. **Lowercase Filtering**: Convert words to lowercase, except for proper nouns.
7. **Stop Word Removal**: Remove stop words from the text.
8. **Lemmatization**: Lemmatize words (reduce them to their base form).

These preprocessing steps are implemented in the `preTraitement` function, which allows for selectively applying the desired preprocessing techniques.

## Data Transformation

After preprocessing, the text data is transformed into a term-frequency-inverse document frequency (TF-IDF) matrix, which represents the importance of words in the corpus.

## Classification

The project explores three different classification tasks:

1. **True vs. False**: Distinguish between genuine news articles (`true`) and fake news articles (`false`).
2. **True or False vs. Other**: Classify articles as either `true` or `false` news, or as `other` (unclassified).
3. **True vs. False vs. Other vs. Mixture**: Classify articles into one of the four classes: `true`, `false`, `other`, or `mixture`.

Several classification algorithms are tested, including Random Forest, Logistic Regression, Naive Bayes, and Decision Trees. The performance of these classifiers is evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Results

The best results are obtained for the "True vs. False" classification task, with an accuracy of 74.7%, precision of 73%, recall of 74%, and F1-score of 73%.

The performance for the "True or False vs. Other" and "True vs. False vs. Other vs. Mixture" tasks is significantly lower, with an accuracy of 61.3% and 37.1%, respectively. This is mainly due to the limited amount of data available for the "other" and "mixture" classes.

## Usage

To run the project, follow these steps:

1. Install the required Python packages (`pandas`, `scikit-learn`, `nltk`, etc.).
2. Execute the Jupyter Notebook.

The project includes a pipeline that performs data preprocessing, transformation, model training, and testing on the provided datasets.

## Future Improvements

Some potential improvements for this project include:

- Exploring advanced techniques for handling class imbalance, such as oversampling or data augmentation.
- Experimenting with different text preprocessing techniques or feature engineering methods.
- Investigating the use of deep learning models or transfer learning for text classification.
- Collecting more data for the minority classes ("other" and "mixture") to improve the performance on those tasks.

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
