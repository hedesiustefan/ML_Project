import re
from typing import Tuple


OFFENSIVE_PATTERNS = [
    # Violence & crime
    r"\b(kill|murder|assassinate|execute|stab|shoot|bomb|explosive|terror|attack)\b",

    # Sexual violence / explicit content
    r"\b(rape|sexual assault|child abuse|porn|explicit|incest)\b",

    # Hate & harassment
    r"\b(hate|racist|racism|nazi|genocide|ethnic cleansing|slur)\b",

    # Self-harm
    r"\b(suicide|self-harm|kill myself|end my life)\b",

    # Illegal activities
    r"\b(drugs|cocaine|heroin|meth|trafficking|weapon|gun|firearm)\b",
]


def violates_content_policy(text: str) -> bool:
    text = text.lower()
    return any(re.search(pattern, text) for pattern in OFFENSIVE_PATTERNS)



MALICIOUS_INTENT_PATTERNS = [
    r"\b(how to make|how to build|how to hack|how to exploit)\b",
    r"\b(bypass|circumvent|evade|avoid detection)\b",
    r"\b(illegal|undetectable|anonymous crime)\b",
]


def has_malicious_intent(text: str) -> bool:
    text = text.lower()
    return any(re.search(pattern, text) for pattern in MALICIOUS_INTENT_PATTERNS)



DOMAIN_KEYWORDS = [
    # --------------------------------------------------
    # General Machine Learning
    # --------------------------------------------------
    "machine learning", "ml", "artificial intelligence", "ai",
    "model", "training", "testing", "validation", "dataset", "data",
    "feature", "label", "input", "output",

    # --------------------------------------------------
    # Optimization & Loss
    # --------------------------------------------------
    "loss", "loss function", "cost function", "objective",
    "optimization", "optimizer", "gradient", "gradient descent",
    "stochastic gradient descent", "sgd", "adam", "momentum",
    "learning rate", "convergence",

    # --------------------------------------------------
    # Statistics & Math Foundations
    # --------------------------------------------------
    "probability", "statistics", "distribution", "expectation",
    "variance", "bias", "error", "noise",
    "linear algebra", "matrix", "vector", "dot product",
    "norm", "distance", "cosine similarity",

    # --------------------------------------------------
    # Classical ML Algorithms
    # --------------------------------------------------
    "regression", "linear regression", "logistic regression",
    "classification", "clustering", "k-means", "svm",
    "decision tree", "random forest", "naive bayes",
    "knn", "nearest neighbor",

    # --------------------------------------------------
    # Learning Paradigms
    # --------------------------------------------------
    "supervised learning", "unsupervised learning",
    "semi-supervised", "reinforcement learning",
    "policy", "reward", "agent", "environment",

    # --------------------------------------------------
    # Neural Networks & Deep Learning
    # --------------------------------------------------
    "neural network", "deep learning", "deep neural network",
    "layer", "hidden layer", "weights", "bias",
    "activation", "relu", "sigmoid", "tanh", "softmax",
    "backpropagation", "forward pass", "backward pass",
    "epoch", "batch", "mini-batch",

    # --------------------------------------------------
    # Model Evaluation & Generalization
    # --------------------------------------------------
    "accuracy", "precision", "recall", "f1 score", "roc",
    "auc", "confusion matrix",
    "overfitting", "underfitting", "generalization",
    "cross-validation", "regularization", "dropout",

    # --------------------------------------------------
    # NLP & Language Models
    # --------------------------------------------------
    "natural language processing", "nlp",
    "token", "tokenization", "vocabulary",
    "embedding", "word embedding", "sentence embedding",
    "context", "sequence", "language model",

    # --------------------------------------------------
    # Transformers & Attention
    # --------------------------------------------------
    "transformer", "attention", "self-attention",
    "encoder", "decoder", "positional encoding",
    "pretraining", "fine-tuning",

    # --------------------------------------------------
    # Retrieval-Augmented Generation (RAG)
    # --------------------------------------------------
    "retrieval", "retriever", "rag", "retrieval-augmented",
    "vector search", "similarity search",
    "faiss", "vector database", "embedding space",

    # --------------------------------------------------
    # Frameworks & Tooling (from literature)
    # --------------------------------------------------
    "pytorch", "tensorflow", "keras",
    "huggingface", "transformers",
    "training loop", "inference",

    # --------------------------------------------------
    # Deployment & Systems (lightweight, relevant)
    # --------------------------------------------------
    "pipeline", "architecture", "model serving",
    "latency", "inference time", "scalability",
]


def is_off_topic(text: str) -> bool:
    text = text.lower()

    # Require at least one strong domain keyword
    if not any(keyword in text for keyword in DOMAIN_KEYWORDS):
        return True

    return False



def input_guardrail(question: str) -> Tuple[bool, str | None]:
    if violates_content_policy(question):
        return False, "The question violates the content policy."

    if has_malicious_intent(question):
        return False, "The question appears to involve malicious or unsafe intent."

    if is_off_topic(question):
        return False, "The question is not related to the course domain or documents."

    return True, None



def output_guardrail(answer: str) -> Tuple[bool, str | None]:
    if violates_content_policy(answer):
        return False, "The generated answer violates the content policy."

    return True, None
