from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200", verify_certs=False)

try:
    print("Pinging Elasticsearch...")
    print("Ping result:", es.ping())
    print("Info:", es.info())
except Exception as e:
    print("Error:", repr(e))
