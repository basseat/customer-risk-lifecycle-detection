"""
Run the Customer Risk Lifecycle detection pipeline against PostgreSQL.
Usage: python3 run_detection.py
Requires: a running PostgreSQL instance with accounts and transactions
          tables loaded (see setup_database.sh), and psycopg2-binary
          and sqlalchemy installed.
"""
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

DB_URI = "postgresql+psycopg2://postgres:postgres@localhost:5432/crl_risk_lifecycle"

conn = psycopg2.connect(
    dbname="crl_risk_lifecycle",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5432,
)
cur = conn.cursor()

with open("detection_queries.sql") as f:
    cur.execute(f.read())
conn.commit()
cur.close()
conn.close()

engine = create_engine(DB_URI)
print("=== Top 20 flagged accounts ===")
flagged = pd.read_sql("SELECT * FROM structuring_likelihood_score LIMIT 20;", engine)
print(flagged.to_string())

try:
    truth = pd.read_csv("seeded_cluster_ground_truth.csv")
    cluster_true = set(truth.account_id)
    flagged_top20 = set(flagged.head(20).account_id)
    recall = len(cluster_true & flagged_top20) / len(cluster_true)
    print(f"\nSeeded cluster size: {len(cluster_true)}")
    print(f"Recovered in top-20: {len(cluster_true & flagged_top20)} ({recall:.0%} recall)")
    print(f"Precision of top-20: {len(cluster_true & flagged_top20)/20:.0%}")
except FileNotFoundError:
    pass

