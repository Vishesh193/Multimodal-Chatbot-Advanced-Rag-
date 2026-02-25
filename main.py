#!/usr/bin/env python3
# ============================================================
# main.py — Entry Point & Demo for Multimodal RAG Chatbot
# ============================================================
#
# Usage:
#   python main.py                        # Interactive chat
#   python main.py --pdf path/to/doc.pdf  # Ingest + chat
#   python main.py --image path/img.jpg   # Direct image chat
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
    print("\n💬 Chat started. Type 'quit' to exit, 'status' for system info.")
    print("   Tip: You can query direct images via: image <path_to_image> <question>\n")
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

        if user_input.lower().endswith(".pdf") and not user_input.lower().startswith("ingest "):
            print(f"\n💡 Tip: It looks like you're trying to ingest a PDF. Please use the 'ingest' command:")
            print(f"   ingest {user_input}")
            continue

        if user_input.lower().startswith("ingest "):
            pdf_path = user_input[7:].strip()
            if not os.path.exists(pdf_path):
                print(f"\n❌ PDF not found: {pdf_path}")
                continue
            try:
                doc_id = rag.ingest_document(pdf_path)
                print(f"\n✅ Ingested! Document ID: {doc_id}")
            except Exception as e:
                print(f"\n❌ Ingestion failed: {e}")
            continue

        # Handle direct image query
        explicit_image_path = None
        query_text = user_input

        if user_input.lower().startswith("image "):
            parts = user_input.split(" ", maxsplit=2)
            if len(parts) >= 3:
                explicit_image_path = parts[1].strip()
                query_text = parts[2].strip()
                if not os.path.exists(explicit_image_path):
                    print(f"\n❌ Image not found: {explicit_image_path}")
                    continue
            else:
                print("\n❌ Invalid format. Use: image <path_to_image> <your question>")
                continue

        # Regular query
        prefix = "🤖 Assistant (Vision): " if explicit_image_path else "🤖 Assistant: "
        print(f"\n{prefix}", end="", flush=True)
        result = rag.query(query_text, explicit_image_path=explicit_image_path)

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
        "--image",
        type=str,
        help="Path to direct image to chat with",
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
        if not rag.indexed_documents and not args.image:
            print("⚠️  No documents indexed. Use --pdf to ingest first.")
        else:
            run_demo(rag)
    else:
        if not rag.indexed_documents and not args.image:
            print(
                "\n💡 Tip: No documents loaded yet.\n"
                "   Type: ingest /path/to/file.pdf\n"
                "   Type: image /path/to/img.jpg What is this?\n"
                "   Or restart with: python main.py --pdf file.pdf\n"
            )
        
        if args.image:
            if not Path(args.image).exists():
                print(f"❌ Image not found: {args.image}")
                sys.exit(1)
            print(f"\n🔍 Analyzing Explicit Image: {args.image}")
            # If the user passed --image, let's ask for the first query right away
            while True:
                q = input("\n🧑 Ask about the image: ").strip()
                if q: break
            print("\n🤖 Assistant (Vision): ", end="", flush=True)
            res = rag.query(q, explicit_image_path=args.image)
            print(res.get("answer", "No answer generated"))
            print("-" * 60)

        interactive_chat(rag)


if __name__ == "__main__":
    main()
