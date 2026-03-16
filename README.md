# Arabic Poetry Assistant

An AI-powered Arabic poetry assistant that can:

1. Search for semantically relevant Arabic poetry verses.
2. Generate a new poetic line starting from a given Arabic word or phrase.

The system combines semantic search with text generation using modern NLP models.

---

## Features

- Arabic verse search using semantic embeddings
- Arabic poetry generation using a GPT-2 based model
- Verse quality filtering and scoring system
- Fast verse retrieval using cached embeddings

---

## Technologies Used

- Python
- Hugging Face Transformers
- Sentence Transformers
- PyTorch
- Pandas / NumPy

---

## How It Works

The system has two main components:

### 1. Verse Finder

The system loads an Arabic poetry dataset and converts each verse into vector embeddings using a multilingual sentence transformer model.

When a user searches for a theme or word, the system retrieves the most semantically similar verses.

### 2. Verse Generator

The generator uses a fine-tuned Arabic GPT-2 poetry model to produce a poetic line starting with the user's input word.

The system then:

- extracts candidate lines
- filters low-quality outputs
- scores each verse
- returns the best result

---

## Example

    examplePhoto.png examplePhoto1.png
--- 

## Dataset

The project uses an Arabic poetry dataset stored as: Arabic_poetry_dataset.csv
the attached file is zipped file need to extract it.

Each poem is split into individual verses and filtered based on quality rules.

---

## Installation

Install required libraries:

pip install transformers torch sentence-transformers pandas numpy

---

## Run the Project 
python main.py

You will see a menu:

1. Find matching verse
2. Generate a verse
3. Exit

---

## Future Improvements

- Improve Arabic meter detection
- Train a custom poetry generation model
- Build a web interface
- Optimize embedding generation speed

---

## Ahmed sayyad 

Computer Engineering graduate interested in AI, NLP and Generative AI.

