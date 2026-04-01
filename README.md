# LLM-Augmented Taxi Analytics Pipeline

This project implements a **Deeply Integrated** LLM-augmented data analytics pipeline using **PySpark** and **LangGraph**, as described in the "LLM-Augmented Data Analytics" framework.

## 🚀 Architecture: Multi-Agent Collaboration

The system uses a **Coordinator-Agent** pattern to manage a machine learning lifecycle. Instead of hard-coded logic, LLM agents observe runtime data statistics and performance metrics to make autonomous decisions.

### Agent Roles:
- **Coordinator**: Orchestrates the workflow and routes state between agents based on the current scenario (`initial_training` vs `march_monitoring`).
- **Data Engineer**: Performs autonomous drift detection by comparing refernce statistics (Jan) against current statistics (Feb).
- **Feature Engineer**: Analyzes dataset schemas to propose transformations (imputation, scaling, encoding).
- **Validator**: Assesses model performance (RMSE, R2) on new data to decide if retraining is necessary.

## 🛠️ Tech Stack
- **Engine**: PySpark (Apache Spark 3.x+)
- **Orchestration**: LangGraph / LangChain
- **LLM**: OpenAI `gpt-4o-mini`
- **Data**: NYC Yellow Taxi Trip Data (sampled)

## 📁 Project Structure
```text
.
├── agents/             # Modular agent logic & Pydantic state models
├── .agents/skills/     # Domain-specific instructions (System Prompts) for each agent
├── main.py             # LangGraph workflow definition and routing logic
├── test_taxi_flow.py   # PySpark orchestrator that invokes the agents
├── setup_mock_data.py  # Utility to fetch and sample NYC Taxi data (MOCK SETUP)
├── setup_inference_demo.py # Trains a mock model & creates demo lag features (MOCK SETUP)
└── inference_pipeline.py  # Final taxi demand inference & ranking logic
```

## 🚥 Quick Start

### 1. Prerequisites
Ensure you have Java (for Spark) and Python 3.10+ installed.

### 2. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_sk_key_here
```

### 4. Run the Pipeline & Inference Demo
First, generate the sampled dataset and inference assets:
```bash
python3 setup_mock_data.py
python3 setup_inference_demo.py
```

Then, run the end-to-end multi-agent flow:
```bash
python3 test_taxi_flow.py
```

And view the driver recommendation system in action:
```bash
python3 inference_pipeline.py
```

## ⚠️ Mock Data & Temporary Files
This project uses **sampled taxi data** (1,000 rows) to keep LLM context small and costs low during the demo. You can safely remove the following when moving to production:
- **`taxi_jan.csv` / `taxi_feb.csv`**: Sampled from TLC NYC Taxi S3 buckets.
- **`march_2016_with_lags.csv`**: Generated mock dataset for the inference demo.
- **`best_model.pkl`**: Mock Random Forest model trained on sampled data.
- **`setup_mock_data.py` / `setup_inference_demo.py`**: Can be replaced by real ETL/Airflow jobs.

## 📊 Pipeline Flow
1. **Scenario 1: Initial Training**
   - Extracts stats for January Taxi data.
   - **Feature Engineer** generates a preprocessing plan.
   - A PySpark `LinearRegression` model is trained.

2. **Scenario 2: March Monitoring**
   - Extracts stats for February Taxi data.
   - Evaluates the January model on February data.
   - **Data Engineer** checks for feature drift.
   - **Validator** checks if R2/RMSE thresholds require a model retrain.
   - **Coordinator** synthesizes findings and terminates the flow.
