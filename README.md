# Redrob-data-Ai
## Getting Started

Follow these steps to set up and run the project.

### 1. Install Dependencies

Install all the required Python libraries using:

```bash
pip install -r requirements.txt
```

---

### 2. Download the AI Model

The project uses the **all-MiniLM-L6-v2** sentence transformer model.

> **Note:** The model files are **not included** in this repository because they exceed GitHub's file size limits

#### Steps

1. Download the model from Hugging Face:
   - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
2. Extract (if necessary) and place the downloaded files in the following directory:

```
models/
└── all-MiniLM-L6-v2/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── model.safetensors
    ├── ...
```

Your project structure should look similar to:

```
project/
│
├── models/
│   └── all-MiniLM-L6-v2/
│
├── data/
├── src/
├── requirements.txt
└── README.md
```

---

## Running a Quick Test

A sample dataset is included with the repository to verify that everything is working correctly.

Run:

```bash
python src/main.py
```

This command uses the included `sample_candidates.json` file and generates sample output.

---

##  Running on the Full Dataset

After obtaining the full candidate dataset, place it inside:

```
data/candidates.jsonl
```

Then run:

```bash
python src/main.py --data data/candidates.jsonl --top_k 100 --total 100000
```

---
