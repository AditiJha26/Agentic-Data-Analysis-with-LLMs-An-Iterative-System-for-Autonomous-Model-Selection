# Agentic Data Analysis with LLMs: An Iterative System for Autonomous Model Selection

## Group Members
- Aditi  
- Mastewal 
- Tucker
- Riker  

---

## Project Overview

This project explores the use of Large Language Models (LLMs) to build an **agentic system** capable of performing end-to-end data analysis on structured tabular datasets.

Unlike traditional machine learning pipelines or single-prompt LLM approaches, our system:
- **Plans** a data analysis strategy
- **Executes** preprocessing and model training
- **Evaluates** model performance
- **Refines** its approach iteratively

The goal is to simulate how a human data analyst would approach a problem — through **reasoning, experimentation, and improvement over time**.

---

## Key Idea

Traditional workflows:
- Manual pipelines → static, require human effort  
- LLMs → generate code once, no iteration  

Our system:
> **LLM + Execution + Feedback Loop = Agentic Data Analyst**

---

## System Architecture

The system consists of the following components:

### 1. Planning Module (LLM)
- Analyzes dataset structure
- Identifies target variable
- Determines task type (classification/regression)
- Suggests preprocessing steps and models

---

### 2. Execution Module
- Applies preprocessing
- Trains machine learning models
- Uses:
  - Logistic Regression
  - Random Forest

---

### 3. Evaluation Module
- Computes performance metrics:
  - Accuracy
  - F1 Score
- Identifies best-performing model

---

### 4. Decision Module (LLM)
- Evaluates model performance
- Decides whether to:
  - STOP (good enough)
  - CONTINUE (needs improvement)

---

### 5. Refinement Module (LLM)
- Updates preprocessing strategy
- Suggests improvements such as:
  - Feature engineering
  - Scaling
  - Better handling of missing values

---

## Pipeline Flow

```text
                ┌──────────────────────┐
                │   Load Dataset       │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │   Summarize Data     │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │   LLM Planner        │
                │ (Initial Plan)       │
                └──────────┬───────────┘
                           ↓
                    ┌──────────────┐
                    │ Train Models │
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │ Evaluate     │
                    └──────┬───────┘
                           ↓
                ┌──────────────────────┐
                │   LLM Decision       │
                └──────┬───────────────┘
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
   ┌──────────────┐        ┌──────────────────┐
   │ STOP         │        │ CONTINUE         │
   └──────────────┘        └─────────┬────────┘
                                      ↓
                           ┌──────────────────────┐
                           │ LLM Refiner          │
                           │ (Update Plan)        │
                           └─────────┬────────────┘
                                     ↓
                                (Loop repeats)

```
---

### Datasets Used

We experimented with structured tabular datasets including:

- Titanic Dataset (classification)
- Credit Risk Dataset
- House Prices Dataset (regression)

---

### Models Used

- Logistic Regression
- Random Forest

---

### Evaluation Metrics

- Accuracy
- F1 Score

---


### Key Features

- Iterative reasoning loop using LLMs
- Dynamic preprocessing based on agent decisions
- Integration of symbolic reasoning (LLM) with numeric computation (ML models)
- Self-refining pipeline

---

### Novel Contribution

This project demonstrates that LLMs can be used not just for code generation, but for:

- Autonomous decision-making
- Iterative improvement
- Closed-loop reasoning systems

We bridge the gap between:

- static ML pipelines
- and intelligent, adaptive systems

---

### Future Work

- Support for more models and advanced pipelines
- Full implementation of feature engineering steps
- Hyperparameter optimization
- Integration with UI (e.g., Streamlit) if not done yet
- Comparison with AutoML systems

---

### Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- OpenAI API
- LangGraph (agent orchestration)

---


### How to Run

- Install dependencies:
    pip install -r requirements.txt
- Add your OpenAI API key in .env:
    OPENAI_API_KEY=your_key_here
- Run the project:
    python src/main.py



---

### Conclusion:

This project successfully demonstrates an agentic data analysis system that can:

- understand datasets
- train models
- evaluate performance
- refine its approach iteratively

It highlights the potential of combining LLMs with execution systems to build autonomous, intelligent pipelines.