#!/usr/bin/env python3
# ============================================================
# main.py — Entry Point & Demo for Multimodal RAG Chatbot
# ============================================================
#
# Usage:
#   python main.py                        # Interactive chat
#   python main.py --pdf path/to/doc.pdf  # Ingest + chat
#   python main.py --demo                 # Run demo queries
# ============================================================

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import RAGConfig, CONFIG
from rag_system import ProductionMultimodalRAG
from utils.logger import get_logger

logger = get_logger(__name__)


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║       🤖 Multimodal RAG Chatbot — Production System      ║
║                                                          ║
║  Stack: HuggingFace + ChromaDB + Groq + Ollama + FAISS   ║
║  Features: HyDE | Multi-Query | Parent-Child | Security  ║
╚══════════════════════════════════════════════════════════╝
""")


def interactive_chat(rag: ProductionMultimodalRAG):
    """Run interactive CLI chat loop."""
    print("\n💬 Chat started. Type 'quit' to exit, 'status' for system info.\n")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break

        if user_input.lower() == "status":
            status = rag.get_status()
            print("\n📊 System Status:")
            print(f"   Documents indexed : {status['indexed_documents']}")
            print(f"   Vector store docs : {status['vector_store'].get('total_documents', 0)}")
            print(f"   Total queries     : {status['total_queries']}")
            print(f"   LLM router        : {status['components']['llm_router']}")
            continue

        if user_input.lower().startswith("ingest "):
            pdf_path = user_input[7:].strip()
            try:
                doc_id = rag.ingest_document(pdf_path)
                print(f"\n✅ Ingested! Document ID: {doc_id}")
            except Exception as e:
                print(f"\n❌ Ingestion failed: {e}")
            continue

        # Regular query
        print("\n🤖 Assistant: ", end="", flush=True)
        result = rag.query(user_input)

        print(result.get("answer", "No answer generated"))

        # Show metadata
        if "sources" in result and result["sources"]:
            print(f"\n   📎 Sources: {len(result['sources'])} chunks retrieved")
            print(f"   ⚡ Time: {result.get('execution_time_s', 0):.2f}s")
            print(f"   🤖 Model: {result.get('model_used', 'unknown')}")

        if result.get("blocked"):
            print(f"\n   🔴 BLOCKED | threat={result.get('threat_level')}")


def run_demo(rag: ProductionMultimodalRAG):
    """Run demonstration queries."""
    print("\n" + "=" * 60)
    print("🎯 DEMO MODE — Running sample queries")
    print("=" * 60)

    demo_queries = [
        "What is the main topic of the document?",
        "Summarize the key findings.",
        "What methods or approaches are described?",
    ]

    for i, query in enumerate(demo_queries, 1):
        print(f"\n[Demo {i}/{len(demo_queries)}]")
        print(f"❓ Query: {query}")

        result = rag.query(query)
        print(f"🤖 Answer: {result.get('answer', 'No answer')[:300]}...")
        print(f"   ⚡ {result.get('execution_time_s', 0):.2f}s | "
              f"🤖 {result.get('model_used', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal RAG Chatbot"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to PDF to ingest before starting chat",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo queries instead of interactive chat",
    )
    parser.add_argument(
        "--groq-key",
        type=str,
        default=os.getenv("GROQ_API_KEY", ""),
        help="Groq API key (or set GROQ_API_KEY env var)",
    )
    args = parser.parse_args()

    print_banner()

    # Build config
    config = CONFIG
    if args.groq_key:
        config.groq_api_key = args.groq_key

    # Initialize system
    logger.info("Initializing RAG system...")
    rag = ProductionMultimodalRAG(config=config)

    # Ingest PDF if provided
    if args.pdf:
        if not Path(args.pdf).exists():
            print(f"❌ PDF not found: {args.pdf}")
            sys.exit(1)
        print(f"\n📄 Ingesting: {args.pdf}")
        doc_id = rag.ingest_document(args.pdf)
        print(f"✅ Ingested | doc_id={doc_id}")

    # Run demo or interactive
    if args.demo:
        if not rag.indexed_documents:
            print("⚠️  No documents indexed. Use --pdf to ingest first.")
        else:
            run_demo(rag)
    else:
        if not rag.indexed_documents:
            print(
                "\n💡 Tip: No documents loaded yet.\n"
                "   Type: ingest /path/to/file.pdf\n"
                "   Or restart with: python main.py --pdf file.pdf\n"
            )
        interactive_chat(rag)


if __name__ == "__main__":
    main()
