# AI-Powered Data Assistant (RAG for Structured Data)

This project is a learning-oriented Proof of Concept (POC) to explore how AI can be integrated with structured data systems using Retrieval-Augmented Generation (RAG).

## ✅ Problem
Business users rely heavily on data teams to run queries, which slows down decision-making and prevents self-service analytics.

## ✅ Solution
I built an AI Data Assistant that:
- Understands natural language questions
- Retrieves relevant rows from a dataset using embeddings + FAISS
- Uses a language model (OpenAI/Azure OpenAI) to generate accurate answers

## ✅ Features
- Natural language querying
- RAG workflow
- Structured data → embeddings → LLM answer
- Lightweight Python implementation
- Runs fully on local machine (except OpenAI API)

## ✅ Tech Stack
- Python  
- Pandas  
- FAISS  
- OpenAI / Azure OpenAI  
- Basic prompt engineering  

## ✅ How to Run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"