# Fine-Tuning Gemma 2B for Code Explanation

This project contains the script and documentation for fine-tuning Google's Gemma 2B instruction-tuned model to become a specialized Python code explainer. The model is trained on the Code Alpaca 20k dataset using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.

The entire process, from data preparation to training and logging, is encapsulated in a single script designed to run in a cloud environment like Kaggle Notebooks.

---
## 🚀 Final Model

The fine-tuned model adapter, which contains the specialized knowledge for explaining Python code, is published on the Hugging Face Hub and is publicly accessible:

* **[mohan1201/gemma-2b-code-explainer](https://huggingface.co/mohan1201/gemma-code-explainer)** 
## 📊 Experiment Tracking

The training process was logged using Weights & Biases. The full experiment, including loss curves, system metrics, and configuration, can be viewed on the public dashboard:

* **[W&B Project Dashboard](https://wandb.ai/mohanchandaka2005-student/huggingface/runs/h6v7jgqe)** 

---
## 🛠️ Tech Stack

* **Model**: Google Gemma 2B-it
* **Dataset**: Code Alpaca 20k
* **Frameworks**: PyTorch, Hugging Face Transformers
* **Fine-Tuning**: PEFT (LoRA), `bitsandbytes` (4-bit Quantization)
* **Experiment Tracking**: Weights & Biases (wandb)

---
## ⚙️ Setup and Training

This project is designed to be run in a cloud environment with a GPU (such as Kaggle or Google Colab).

### 1. Prerequisites
* A Kaggle or Google Colab account.
* API keys for Hugging Face (with `write` permissions) and Weights & Biases.

### 2. Environment Setup
* Create a new notebook and select a GPU accelerator (e.g., T4).
* Add your `HF_TOKEN` and `WANDB_API_KEY` as secrets.
* Add the [Code Alpaca 20k dataset](https://www.kaggle.com/datasets/plameneduardo/code-alpaca-20k) as a data source for the notebook.

### 3. Run the Training
The fine-tuning process is handled by a single script (`train.py` in this repository, or a single cell in the notebook). It performs all necessary steps:
1. Installs and imports all required libraries.
2. Logs in to Hugging Face and W&B using the provided secrets.
3. Loads the base Gemma 2B model with 4-bit quantization.
4. Loads and formats the Code Alpaca dataset.
5. Configures and runs the fine-tuning job using the `transformers.Trainer`.
6. Saves the final LoRA adapter and pushes it to the Hugging Face Hub.

### 4. Inference
To use the trained model, you can load the base Gemma 2B model and then apply the fine-tuned adapter from the Hugging Face Hub. An example inference script is available in the repository.
