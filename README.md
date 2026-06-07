# Phenotyping with AI Agents using the Iowa Gambling Task

## Overview

This project investigates behavioural phenotypes in AI agents by analysing decision-making patterns in the Iowa Gambling Task (IGT). Large Language Models (LLMs) (DeepSeek and Gemini) are treated as participants and compared against human data.

The pipeline consists of:

1. Generating behavioural data from AI agents
2. Processing and structuring datasets
3. Fitting behavioural models (Rescorla-Wagner)
4. Clustering and analysing behavioural patterns

---

## Setup Requirements

### Python dependencies

Create a virtual environment and install pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you want to run notebooks:

```bash
pip install jupyter
```

### API Keys

You will need API access for:

- DeepSeek
- Gemini

Set your API keys as environment variables:

```bash
export DEEPSEEK_API_KEY="your_key_here"
export GEMINI_API_KEY="your_key_here"
```

---

## Project Structure

Key Folders:
```
src/                data generation and modelling scripts  
data/               processed datasets  
notebooks/          analysis and visualisation  
docs/               report and supporting documents  
```

---

## Pipeline
Note: {model} is deepseek/ gemini

### 1. Generate Model Data

Navigate to:
```bash
src/{model}/game_play
```

Run all gameplay scripts (equivalent scripts for gemini):
```bash
python3 deepseek_IOWA_stop_100_trials.py
python3 deepseek_IOWA_stop_max_profit.py
python3 deepseek_IOWA_2500_profit.py
python3 deepseek_IOWA_PERSONAS.py
```
This generates raw gameplay CSVs in:
```bash
src/{model}/game_play
```

### 2. Process and Organise Data

From the src/ directory, run:
```bash
python3 process_igt_data.py
```

This will:

- Add metadata (variant, persona, model)
- Combine datasets
- Move processed files to:
```bash
data/iowa_models/{model}/
```

### 3. Run Behavioural Modelling (Rescorla-Wagner)
**Model Data:**
From:
```bash
src/{model}/
```

Run:
```bash
python3 {model}_RW.py
python3 {model}_RW_personas.py
```

This produces:
- RW parameters (alpha, beta)
- Persona-specific results

**Human Data:**
From:
```bash
src/human/
```
Run:
```bash
python3 human_RW.py
```

This produces:
- RW parameters for human participants
- Outputs saved in:
```bash
data/iowa_human/
```

### 4. Human Data Processing
Human data is already included in:
```bash
data/iowa_human/
```

Scripts in:
```bash
src/human/
```
can be used if preprocessing is required

### 5. Data Validation
Run the notebook:
```bash
notebooks/Data Cleaning.ipynb
```
This checks:
- Missing values
- Dataset consistency
- Correct formatting

### 6. Analysis and Visualisation
Use notebooks in:
```bash
notebooks/
```

**Key notebooks:**

- [Behavioural Analysis](notebooks/Behavioural%20Analysis.ipynb)  
- [Behavioural Features Clustering](notebooks/Behavioural%20Features%20Clustering.ipynb)  
- [RW Modelling](notebooks/RW%20Modeling.ipynb)  
- [RW Parameters Clustering](notebooks/RW%20Parameters%20Clustering.ipynb)

These perform:
Feature extraction
PCA visualisation
Clustering (e.g. K-means)
Model comparison

---

## Outputs
Final datasets include:

**Model Data**
```bash
data/iowa_models/{model}/
```
- *_igt_combined.csv
- *_igt_persona_combined.csv
- *_rw_results.csv

**Human Data**
```bash
data/iowa_human/
```

---

## Notes
- Ensure API quotas are sufficient before running large experiments
- Persona experiments may be computationally expensive
- Processed data will overwrite previous outputs when rerun
- Raw gameplay files are moved or archived after processing

---

## Summary 
To fully reproduce results:
1. Generate gameplay data
2. Run processing script
3. Fit Rescorla–Wagner models (models and human)
4. Validate data
5. Run notebooks for analysis

---
## Acknowledgments
We would like to thank our supervisor, Tomas ward for his guidance and support throughout this project and his research assistant, Harmonia Wang for sharing insights and advice on phenotyping research.

## Authors

Sarah O'Hanlon (sarah.ohanlon24@mail.dcu.ie)
Tanuli Liyanage (tanuli.damburaliyanage2@mail.dcu.ie)

---