# 🍽️ Restaurant Search NER Fine-Tuning

---

## 🧠 Project Overview

This project focuses on **fine-tuning DistilBERT** for **Named Entity Recognition (NER)** tasks to identify restaurant-related entities from user search queries such as:

> *“5 star restaurants in my town”*

The goal is to build a lightweight model capable of understanding real-world restaurant search patterns and classifying entities like **cuisine**, **location**, and **rating**.

---

## ⚙️ Frameworks & Libraries Used

* **Python** 🐍
* **PyTorch** ⚡
* **Hugging Face Transformers** 🤗
* **Datasets** 🧾
* **Scikit-learn** 📊
* **Pandas** 🧮
* **Matplotlib** 📈
* **Google Colab** 💻

---

## ✨ Key Features

* Fine-tuned **DistilBERT** for restaurant-based NER.
* Custom tokenizer alignment to handle word-piece tokens.
* Evaluation with precision, recall, and F1 metrics.
* Dataset preprocessing and label alignment function for consistent tagging.
* Logging disabled to avoid unwanted **wandb** initialization issues.

---

## 🧩 Label Alignment Example

Example input:

```
['[CLS]', '5', 'star', 'rest', '##ura', '##nts', 'in', 'my', 'town', '[SEP]']
```

### 🧠 Solution:

```python
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
```

---

## 🚀 Future Plans

* Deploy model as an **interactive Streamlit app** for real-time restaurant entity recognition.
* Integrate model into a **chat-based restaurant recommendation system**.

---

## ⚡ Challenges Faced

| Challenge                     | Description                                         | Solution                                                   |
| ----------------------------- | --------------------------------------------------- | ---------------------------------------------------------- |
| ⏱️ Long Training Time         | Training DistilBERT takes time on limited hardware. | Used smaller batch sizes and gradient accumulation.        |
| 💻 Colab Free GPU Limitations | Frequent disconnections or resource limits.         | Regular checkpoints and saving model to Google Drive.      |
| 🧩 Label Misalignment         | Tokens split incorrectly, causing tag mismatches.   | Implemented custom `tokenize_and_align_labels()` function. |
| 🧰 W&B Errors                 | `wandb.init()` errors appeared during training.     | Disabled W&B logging using environment variables.          |

---

## 🏁 Outcome

Successfully fine-tuned a **DistilBERT** model for restaurant search query NER, achieving efficient inference while maintaining accuracy.
