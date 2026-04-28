import math
from collections import defaultdict, Counter


class NaiveBayesSpamClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = {}
        self.word_probs = {}
        self.vocab = set()
        self.class_word_counts = {}
        self.class_total_words = {}
        self.classes = []

    def tokenize(self, text):
        return str(text).split()

    def fit(self, X_train, y_train):
        """
        Train Multinomial Naive Bayes from scratch.

        Parameters
        ----------
        X_train : list[str]
            Cleaned text samples.
        y_train : list[int]
            Labels, where 0 = Not Spam and 1 = Spam.
        """
        self.classes = sorted(set(y_train))

        total_docs = len(y_train)
        class_doc_counts = Counter(y_train)

        # P(class)
        for c in self.classes:
            self.class_priors[c] = class_doc_counts[c] / total_docs

        # Count words for each class
        self.class_word_counts = {
            c: defaultdict(int) for c in self.classes
        }

        self.class_total_words = {
            c: 0 for c in self.classes
        }

        for text, label in zip(X_train, y_train):
            words = self.tokenize(text)

            for word in words:
                self.vocab.add(word)
                self.class_word_counts[label][word] += 1
                self.class_total_words[label] += 1

        vocab_size = len(self.vocab)

        # P(word | class) with Laplace smoothing
        self.word_probs = {
            c: {} for c in self.classes
        }

        for c in self.classes:
            total_words = self.class_total_words[c]

            for word in self.vocab:
                word_count = self.class_word_counts[c][word]

                probability = (
                    (word_count + self.alpha)
                    / (total_words + self.alpha * vocab_size)
                )

                self.word_probs[c][word] = probability

    def predict_one(self, text):
        """
        Predict one text sample.

        Returns
        -------
        int
            0 = Not Spam, 1 = Spam
        """
        words = self.tokenize(text)
        vocab_size = len(self.vocab)

        scores = {}

        for c in self.classes:
            score = math.log(self.class_priors[c])
            total_words = self.class_total_words[c]

            unknown_word_prob = (
                self.alpha / (total_words + self.alpha * vocab_size)
            )

            for word in words:
                if word in self.vocab:
                    score += math.log(self.word_probs[c][word])
                else:
                    score += math.log(unknown_word_prob)

            scores[c] = score

        return max(scores, key=scores.get)

    def predict(self, X):
        """
        Predict multiple text samples.
        """
        return [self.predict_one(text) for text in X]

    def predict_proba_one(self, text):
        """
        Return probability for one text sample.

        Returns
        -------
        dict
            Example: {0: 0.91, 1: 0.09}
        """
        words = self.tokenize(text)
        vocab_size = len(self.vocab)

        log_scores = {}

        for c in self.classes:
            score = math.log(self.class_priors[c])
            total_words = self.class_total_words[c]

            unknown_word_prob = (
                self.alpha / (total_words + self.alpha * vocab_size)
            )

            for word in words:
                if word in self.vocab:
                    score += math.log(self.word_probs[c][word])
                else:
                    score += math.log(unknown_word_prob)

            log_scores[c] = score

        # Convert log scores to probabilities safely
        max_log = max(log_scores.values())

        exp_scores = {
            c: math.exp(log_scores[c] - max_log)
            for c in self.classes
        }

        total = sum(exp_scores.values())

        probabilities = {
            c: exp_scores[c] / total
            for c in self.classes
        }

        return probabilities