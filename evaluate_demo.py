"""
evaluate_demo.py — RAG Evaluation Demo
========================================
Demonstrates how to use the evaluation module with a sample RAG pipeline output.

Run:
    python evaluate_demo.py
"""

from evaluation import RAGEvaluator

# ── Sample data ───────────────────────────────────────────────────────────
#
# In a real scenario, these would come from your RAG pipeline:
#   query                  → user question
#   retrieved_documents    → content of chunks returned by the retriever
#   ground_truth_documents → the correct/relevant chunks (for retrieval metrics)
#   generated_answer       → LLM response
#   reference_answer       → human-written or gold-standard answer

sample = {
    "query": "What is Retrieval-Augmented Generation (RAG) and how does it work?",

    "retrieved_documents": [
        # Chunk 1 — from the actual paper
        "Retrieval-Augmented Generation (RAG) is a technique that combines "
        "a retrieval component with a generative language model. Given a query, "
        "the retrieval module fetches relevant documents from a knowledge base, "
        "and the generator conditions on both the query and retrieved documents "
        "to produce a response.",

        # Chunk 2 — additional context
        "Unlike pure generative models that rely solely on parametric knowledge, "
        "RAG systems have access to an external, updatable knowledge store. "
        "This reduces hallucinations and allows the model to cite specific sources.",

        # Chunk 3 — slightly off-topic (tests faithfulness)
        "Large language models such as GPT-4 and LLaMA 3 are trained on "
        "trillions of tokens and can generate coherent, fluent text across "
        "many domains.",
    ],

    "ground_truth_documents": [
        # The truly relevant chunks (used to compute Precision/Recall)
        "Retrieval-Augmented Generation (RAG) is a technique that combines "
        "a retrieval component with a generative language model. Given a query, "
        "the retrieval module fetches relevant documents from a knowledge base, "
        "and the generator conditions on both the query and retrieved documents "
        "to produce a response.",

        "Unlike pure generative models that rely solely on parametric knowledge, "
        "RAG systems have access to an external, updatable knowledge store. "
        "This reduces hallucinations and allows the model to cite specific sources.",
    ],

    "generated_answer": (
        "Retrieval-Augmented Generation, or RAG, is a framework that enhances "
        "large language models by pairing them with a retrieval system. "
        "When a user submits a query, the retriever searches an external "
        "knowledge base to find relevant document chunks. These chunks are then "
        "included in the prompt alongside the query, allowing the generator to "
        "produce a more accurate and grounded response. RAG reduces hallucinations "
        "because the model can reference specific source documents rather than "
        "relying solely on its training data."
    ),

    "reference_answer": (
        "RAG (Retrieval-Augmented Generation) is a hybrid approach that combines "
        "a dense retrieval module with a sequence-to-sequence generator. "
        "For a given input query, the retriever fetches the top-K relevant "
        "passages from a large document corpus. The generator then conditions "
        "on the concatenation of the query and retrieved passages to produce "
        "the final answer. This approach grounds the model's output in external "
        "knowledge, improving factual accuracy and reducing hallucinations compared "
        "to purely parametric models."
    ),
}

# ─────────────────────────────────────────────────────────────────────────
# Initialise evaluator (no LLM client → heuristic fallbacks will be used)
# To use LLM-as-Judge, pass your Groq or Ollama client:
#   from llm.clients import GroqClient
#   groq = GroqClient(api_key="...", model="llama-3.3-70b-versatile")
#   evaluator = RAGEvaluator(llm_client=groq)
# ─────────────────────────────────────────────────────────────────────────
evaluator = RAGEvaluator(llm_client=None)

report = evaluator.evaluate(
    query                  = sample["query"],
    retrieved_documents    = sample["retrieved_documents"],
    ground_truth_documents = sample["ground_truth_documents"],
    generated_answer       = sample["generated_answer"],
    reference_answer       = sample["reference_answer"],
    k                      = 3,
)

# Print the human-readable summary
print(report.summary())
print(f"\n⭐ Overall Score: {report.overall_score():.4f}")

# Optionally save the full JSON report
# report.to_json("eval_report.json")
