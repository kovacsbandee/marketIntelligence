# Elasticsearch Setup and Python Integration Guide

This guide will help you install Elasticsearch on Ubuntu 22.04 and integrate it into your Python project for storing and searching news data, such as that from the Alpha Vantage API.

---

## 🛠️ Part 1: Install Elasticsearch on Ubuntu 22.04

### 1. Install Required Packages
```bash
sudo apt update
sudo apt install apt-transport-https ca-certificates curl gnupg
```

### 2. Add Elasticsearch GPG Key
```bash
curl -fsSL https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo gpg --dearmor -o /usr/share/keyrings/elasticsearch-keyring.gpg
```

### 3. Add Elasticsearch Repository
```bash
echo "deb [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-8.x.list
```

### 4. Install Elasticsearch
```bash
sudo apt update
sudo apt install elasticsearch
```

### 5. Enable and Start Elasticsearch
```bash
sudo systemctl enable elasticsearch
sudo systemctl start elasticsearch
```

### 6. Test Installation
```bash
curl -X GET "localhost:9200"
```
You should see a JSON response with version information.

---

## 🐍 Part 2: Set Up Elasticsearch Python Client

### 1. Install Python Package
```bash
pip install elasticsearch
```

### 2. Basic Python Integration Script
```python
from elasticsearch import Elasticsearch

# Connect to local Elasticsearch instance
es = Elasticsearch("http://localhost:9200")

# Index sample news article
doc = {
    "title": "Tech stocks rally as inflation slows",
    "symbol": "AAPL",
    "source": "AlphaVantage",
    "timestamp": "2025-07-12T08:00:00",
    "summary": "Apple and other tech stocks are climbing as inflation data improves."
}

# Index the document
res = es.index(index="news", document=doc)
print(f"Document indexed with ID: {res['_id']}")

# Search for news by keyword
query = {
    "query": {
        "match": {
            "summary": "inflation"
        }
    }
}
results = es.search(index="news", body=query)
for hit in results["hits"]["hits"]:
    print(f"Title: {hit['_source']['title']}, Score: {hit['_score']}")
```

---

## 🗃️ Optional: Create a Custom Index Mapping

```python
index_config = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "symbol": {"type": "keyword"},
            "source": {"type": "keyword"},
            "timestamp": {"type": "date"},
            "summary": {"type": "text"}
        }
    }
}

# Create index (only once)
es.indices.create(index="news", body=index_config)
```

---

## ✅ What's Next?

- Store Alpha Vantage articles into the `news` index.
- Add fields like sentiment, categories, and tickers.
- Build time-based aggregations or visualizations (e.g., with Kibana).

---

> You can save this guide alongside your project as `elasticsearch_setup.md` for future reference.

