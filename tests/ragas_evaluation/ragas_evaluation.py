import os
import time
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Import the production workflow
from src.graph import build_rag_graph

# ==========================================

# 1. CONFIGURATION & LLM SETUP
# ==========================================
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")

# Using Mixtral for evaluation to provide a neutral "judge" perspective
evaluator_llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)

# Embeddings used for context and answer relevancy calculations
evaluator_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the production workflow once
workflow = build_rag_graph()

# ==========================================
# 2. EVALUATION DATASET (30 Questions)
# ==========================================
eval_data = [
    # customer_segments.csv
    {"question": "What is the typical size of Federal Agencies that use Granicus?", "ground_truth": "1000-50000"},
    {"question": "What is the budget range for State Government segments?", "ground_truth": "Medium-High"},
    {"question": "What is the primary use case for County Governments?", "ground_truth": "Public Notifications, Meeting Management, Records Management"},
    {"question": "What is the implementation timeline for Federal Agencies?", "ground_truth": "6-12 months"},
    {"question": "What are the key requirements for State Governments?", "ground_truth": "Scalability, Integration with Existing Systems, Mobile Accessibility"},
    {"question": "Who are the primary decision-makers for County Governments?", "ground_truth": "County Administrator, IT Director, Clerk's Office"},
    
    # faq_content.txt
    {"question": "When was Granicus founded?", "ground_truth": "1999"},
    {"question": "What tools does Granicus's cloud-based platform include?", "ground_truth": "Digital communications, meeting management, public records, and civic engagement."},
    
    # feature_comparison.csv
    {"question": "Does the Starter tier of GovDelivery Communications Cloud include SMS Alerts?", "ground_truth": "No"},
    {"question": "Which tier of GovDelivery Communications Cloud includes Voice Messaging?", "ground_truth": "Enterprise"},
    {"question": "Is email marketing included in all tiers of the GovDelivery Communications Cloud?", "ground_truth": "Yes, it is a core feature across all tiers."},
    
    # granicus_products.html / .md
    {"question": "How many government organizations does Granicus serve worldwide?", "ground_truth": "Over 5,500"},
    {"question": "What is the mission statement of Granicus?", "ground_truth": "To connect governments and their communities."},
    
    # pricing_matrix.csv
    {"question": "What is the monthly price of the Starter tier for GovDelivery Communications Cloud?", "ground_truth": "$500"},
    {"question": "What is the maximum number of subscribers for the Professional tier of GovDelivery Communications Cloud?", "ground_truth": "50000"},
    {"question": "What level of support is provided in the Enterprise tier of GovDelivery Communications Cloud?", "ground_truth": "24/7 Premium Support"},
    {"question": "How much is the setup fee for the Professional tier of GovDelivery Communications Cloud?", "ground_truth": "$500"},
    {"question": "What is the annual price for the Enterprise tier of GovDelivery Communications Cloud?", "ground_truth": "$54000"},
    {"question": "Which tier of GovDelivery Communications Cloud includes a Dedicated Account Manager?", "ground_truth": "Enterprise"},
    
    # release_notes.txt
    {"question": "What new AI feature was introduced in GovDelivery Communications Cloud v4.2.0?", "ground_truth": "Enhanced AI-powered message optimization with natural language suggestions."},
    {"question": "When was GovDelivery Communications Cloud v4.2.0 released?", "ground_truth": "September 1, 2025"},
    {"question": "What collaborative integration was added in GovDelivery Communications Cloud v4.2.0?", "ground_truth": "Integration with Microsoft Teams for collaborative message creation."},
    {"question": "Does the improved mobile app in version 4.2.0 support offline message composition?", "ground_truth": "Yes"},
    {"question": "What custom feature was introduced for endpoints in the 2025 Q3 release?", "ground_truth": "Custom webhook endpoints"},
    {"question": "What kind of advanced subscriber segmentation was added in v4.2.0?", "ground_truth": "Segmentation based on engagement patterns."},
    
    # technical_specs.txt
    {"question": "On which cloud provider is Granicus hosted?", "ground_truth": "Amazon Web Services (AWS)"},
    {"question": "What is the uptime SLA for the Granicus auto-scaling infrastructure?", "ground_truth": "99.9%"},
    {"question": "What type of SOC certification does the Granicus security framework hold?", "ground_truth": "SOC 2 Type II certified"},
    {"question": "What capability does the global content delivery network (CDN) provide for Granicus?", "ground_truth": "Optimal performance"},
    {"question": "What kind of data centers does Granicus use to ensure availability?", "ground_truth": "Redundant data centers with automatic failover capabilities"}
]

# ==========================================
# 3. RAG PIPELINE INTEGRATION
# ==========================================
def run_rag_pipeline(question: str):
    """
    Invokes the production LangGraph workflow to extract the synthesized answer
    and raw context chunks for RAGAS.
    """
    initial_state = {
        "question": question, 
        "documents": [], 
        "metadata_manifest": [],
        "confidence_score": 0.0,
        "latency_ms": 0.0
    }
    
    # Execute workflow
    result = workflow.invoke(initial_state)
    
    # Parse the answer string (which is a JSON string from synthesize_node)
    try:
        answer_data = json.loads(result.get("answer", "{}"))
        synthesized_answer = answer_data.get("synthesized_answer", "No response.")
    except json.JSONDecodeError:
        synthesized_answer = result.get("answer", "No response.")

    # Extract raw text from retrieved LangChain Document objects
    retrieved_docs = result.get("documents", [])
    contexts = [doc.page_content for doc in retrieved_docs]
    
    return {
        "answer": synthesized_answer,
        "contexts": contexts
    }

# ==========================================
# 4. DATA PREPARATION & EVALUATION
# ==========================================
def main():
    print(f"🚀 Starting RAGAS Evaluation on {len(eval_data)} queries using Mixtral-8x7b-32768...")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # 4a. Run the RAG system
    for i, item in enumerate(eval_data):
        print(f"Processing query {i+1}/{len(eval_data)}: {item['question']}")
        time.sleep(3)
        rag_output = run_rag_pipeline(item["question"])
        time.sleep(3)
        questions.append(item["question"])
        answers.append(rag_output["answer"])
        contexts.append(rag_output["contexts"])
        ground_truths.append(item["ground_truth"])
        
        # Prevent rate limit issues
        time.sleep(1)

    # 4b. Format as HuggingFace Dataset
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    hf_dataset = Dataset.from_dict(dataset_dict)

    # 4c. Define Metrics
    metrics = [
        faithfulness,       # Hallucination check
        answer_relevancy,   # Usefulness check
        context_precision,  # Retrieval quality
        context_recall      # Coverage check
    ]

    # 4d. Run Ragas Evaluation
    print("\n📊 Running LLM-as-a-judge evaluation...")
    results = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # 4e. Output Results
    print("\n✅ === EVALUATION RESULTS ===")
    print(results)
    
    # Save to CSV
    df_results = results.to_pandas()
    df_results.to_csv("ragas_evaluation_results.csv", index=False)
    print("\n📁 Detailed results saved to 'ragas_evaluation_results.csv'.")

if __name__ == "__main__":
    main()
