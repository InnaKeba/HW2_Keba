import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import chromadb
from langgraph.types import interrupt, Command

# 1. Перевірка SqliteSaver
try:
    conn = sqlite3.connect(':memory:')
    saver = SqliteSaver(conn)
    print('SqliteSaver: OK')
except Exception as e:
    print(f'SqliteSaver ПОМИЛКА: {e}')

# 2. Перевірка ChromaDB
try:
    client = chromadb.Client()
    collection = client.create_collection('test_collection')
    collection.add(
        documents=['LangGraph - це фреймворк для агентів', 'ChromaDB - vector database'],
        ids=['doc1', 'doc2'],
    )
    results = collection.query(query_texts=['агентний фреймворк'], n_results=1)
    print(f'ChromaDB: OK (Знайдено текст: {results["documents"][0]})')
except Exception as e:
    print(f'ChromaDB ПОМИЛКА: {e}')

# 3. Перевірка interrupt
try:
    print('interrupt: OK')
except Exception as e:
    print(f'interrupt ПОМИЛКА: {e}')