## PyDocs-RAG

This is a project that aims to create a RAG for the python documentation to make using, understanding and navigating the PyDocs easier for both new and experienced users. The focus will be on evaluating the correctness of the answers delivered by the system.  
<!-- A programming language's documentation is the first point of access to it for both beginners andd experienced programmers. However, navigating the documentation to get answers is rarely easy because the docs are designed for reference, not teaching or quick understanding. A RAG would be great to solve this problem. 

A RAG-powered Python Docs Assistant that acts as a smart guide to Python documentation.

1. Ask any Python question in natural language.

2. The system retrieves relevant doc sections from official sources.

3. The LLM explains it in clear, contextual terms — with code examples where possible.

4. Provides citations & links back to the official docs for verification. -->


## How to Run

1. Clone the repo
```
git clone https://github.com/Perseus1702/PyDocs-RAG.git

```

2. Create a venv and install the dependencies
```
python -m .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

3. Run the scraper script to build the knowledge base
```
python scrape_build.py
```
4. Normalize and index the knowledge base
```
python index_kb.py
```
5. Ask away
```
python ask.py --q "How to use else statement"
```