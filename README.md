# Instruction Manual and Documentation for Fine-Tuning and Inference using Unsloth
# IMPORTANT: Make sure to copy this folder to MyDrive before running any notebook. Copy only the All Files folder and not its parent. Code files contain the finetuning as well as zero shot methods. Data files contain the data used (MedMCQA). There is also a notebook for the Data Analysis.
## Overview
This document provides a detailed guide to fine-tuning and performing inference with the Unsloth framework. It covers the data analysis, installation process, dataset preparation, model setup, hyperparameter explanations, and testing methodology. The instructions are applicable across multiple notebooks, with specific differences highlighted per category.

---

## Table of Contents
1. [Data Analysis](#data-analysis)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Model Configuration](#model-configuration)
5. [Training Parameters](#training-parameters)
6. [Inference Workflow](#inference-workflow)
7. [Hyperparameter Details](#hyperparameter-details)
8. [Performance Metrics](#performance-metrics)
9. [Zero Shot Inference](#zero-shot-inference)
10. [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
11. [Results](#results)

---
# 1. Medical MCQ Dataset Exploratory Data Analysis

[This](https://colab.research.google.com/drive/1ccu7-tVucHSCNGCX3Ay5tuaz4bHAWFqi) Jupyter notebook presents a comprehensive exploratory data analysis of the MedMCQA dataset, which contains medical multiple-choice questions. The analysis is aimed at preparing the data for fine-tuning multiple models for question answering tasks.

## Key Components

### Data Loading and Preparation

The notebook begins by importing necessary libraries and loading data from JSON files for train, development, and test sets. It creates pandas DataFrames for each dataset and performs initial analysis on their structure and content.

### Dataset Overview

The analysis provides detailed information about each dataset, including:

- Number of questions
- Number and names of columns
- Subject distribution

A summary DataFrame is created to compare key statistics across train, development, test, and combined datasets.

### Data Exploration

The notebook includes code to display the first few rows of the training dataset, giving insights into the structure of individual questions, answer choices, and associated metadata.

## Code Structure

The notebook is organized into several sections:

1. **Import Required Libraries**: Sets up the necessary Python libraries for data analysis and visualization.

2. **Load and Prepare Data**: Contains functions to load JSON files and create DataFrames. It also includes code to analyze and print dataset statistics.

3. **Exploratory Data Analysis**: Presents the first few rows of the training dataset for a closer look at the data structure.

## Key Findings

- The combined dataset contains 193,155 questions across 21 unique medical subjects.
- There are differences in the number of columns between the test dataset (9 columns) and the train/dev datasets (11 columns).
- The subject distribution varies across datasets, with some subjects like "Medicine" and "Surgery" having higher representation.

## Usage

This notebook serves as a starting point for understanding the MedMCQA dataset structure and composition. It can be used to inform further analysis, feature engineering, and model development for medical question-answering tasks.

## 2. Environment Setup
### Prerequisites
Ensure you have the following:
- Google Colab environment with GPU enabled.
- Python 3.7 or later.
- Installed libraries: `unsloth`, `datasets`, `transformers`, `trl`, and `torch`.

### Installation Commands
```python
%%capture
!pip install unsloth

# Upgrade to the latest nightly build
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

---

## 3. Dataset Preparation
### Dataset Format
- Each dataset is stored as a `.jsonl` file with the following structure:
  ```json
  {
    "instruction": "...",
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "..."
  }
  ```

### Loading the Dataset
Specify the dataset path and load it as follows:
```python
processed_dataset_path = "/content/drive/MyDrive/NLP_Project_105/Datasets/Awesome-Medical-Dataset_MedMCQA/Data/Phi-3.5_Data/processed_medical_data"
category = "Anatomy"
dataset = load_dataset("json", data_files=os.path.join(processed_dataset_path, f"{category}_train_processed.jsonl"), split="train")
```

---

## 4. Model Configuration
### Initializing the Model
Load the model with appropriate configurations:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3.5-mini-instruct",
    max_seq_length=2048,
    dtype=None,  # Auto-detection of precision
    load_in_4bit=True  # Enables 4-bit quantization
)
```

### Applying LoRA Fine-Tuning
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407
)
```

---

## 5. Training Parameters
### Training Arguments
```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=True,  # Use mixed precision
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none"
    )
)
```

---

## 6. Inference Workflow
### Preparing Inputs
```python
def format_input_for_inference(example):
    instruction = example['instruction']
    question = example['question']
    options = " ".join(example['options'])
    return {"from": "human", "value": f"Instruction: {instruction}\nQuestion: {question}\nOptions: {options}"}

inference_inputs = [format_input_for_inference(example) for example in test_dataset]
```

### Generating Outputs
```python
for idx, message in enumerate(messages):
    inputs = tokenizer.apply_chat_template(
        [message],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True)
    generated_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
```

---

## 7. Hyperparameter Details
- **`max_seq_length`**: Defines the maximum sequence length supported by the model.
- **`dtype`**: Determines the precision (e.g., `float16`, `bfloat16`).
- **`load_in_4bit`**: Enables 4-bit quantization for reduced memory usage.
- **`r`**: LoRA rank; controls the number of trainable parameters.
- **`lora_alpha`**: Scaling factor for LoRA updates.
- **`learning_rate`**: Controls the rate of weight updates during training.
- **`gradient_accumulation_steps`**: Accumulates gradients over multiple steps for effective batch size.

---

## 8. Performance Metrics
- **Accuracy Calculation**:
  ```python
  accuracy = sum(result['is_correct'] for result in results) / len(results)
  print(f"Overall Accuracy: {accuracy:.2%}")
  ```
- **Memory Utilization**:
  ```python
  gpu_stats = torch.cuda.get_device_properties(0)
  print(f"GPU: {gpu_stats.name}, Max Memory: {gpu_stats.total_memory / 1e9:.2f} GB")
  ```

---

## 9. Zero Shot Inference
- To run these notebooks, remove the cells from training, and run inference directly after loading the model and mapping the data.

## 10. Common Issues and Troubleshooting
### Issue: Memory Overflow
- **Solution**: Reduce batch size or enable 4-bit quantization.

### Issue: Incorrect Answers
- **Solution**: Verify dataset formatting and ensure proper template mapping.

## 11. Results

| **Subject/Model Accuracy** | **Zero-Shot Phi 3.5-mini-instruct** | **Fine-tuned Phi 3.5-mini-instruct** | **Improvement (%)** |
|-----------------------------|------------------------------------|-------------------------------------|---------------------|
| **Anatomy**                | 39.74%                             | 50%                                 | 25.85%             |
| **Biochemistry**           | 47.37%                             | 67.25%                              | 41.92%             |
| **Gynaecology and Obstetrics** | 36.16%                         | 51.34%                              | 41.95%             |
| **Internal Medicine**      | 39.32%                             | 51.86%                              | 31.93%             |
| **Microbiology**           | 37.70%                             | 52.46%                              | 39.13%             |
| **Pathology**              | 43.03%                             | 57.86%                              | 34.49%             |
| **Pediatrics**             | 40.60%                             | 50.85%                              | 25.21%             |
| **Pharmacology**           | 48.56%                             | 65.43%                              | 34.74%             |
| **Physiology**             | 42.69%                             | 60.23%                              | 41.07%             |
| **Psychiatry**             | 43.75%                             | 56.25%                              | 28.57%             |
| **Surgery**                | 36.59%                             | 50.14%                              | 37.01%             |
| **Overall Accuracy**       | **41.41%**                         | **55.79%**                           | **34.72%**         |

This table highlights the percentage improvement for each subject, showing the significant accuracy gains achieved through fine-tuning.

---

This guide serves as a reference for all notebooks and categories. Ensure to modify dataset paths and category-specific configurations as needed.
