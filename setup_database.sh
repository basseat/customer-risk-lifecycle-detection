#!/bin/bash
# Sets up the crl_risk_lifecycle PostgreSQL database and loads the CSVs.
# Run once, from the folder containing accounts.csv and transactions.csv.
set -e

createdb crl_risk_lifecycle

psql -d crl_risk_lifecycle -c "
CREATE TABLE accounts (
    account_id TEXT PRIMARY KEY,
    initial_risk_category TEXT,
    kyc_documentation_completeness NUMERIC,
    onboarding_channel TEXT,
    onboarding_device_id TEXT,
    opened_at TIMESTAMP
);
CREATE TABLE transactions (
    transaction_id TEXT PRIMARY KEY,
    account_id TEXT REFERENCES accounts(account_id),
    direction TEXT,
    amount NUMERIC,
    counterparty_id TEXT,
    counterparty_country TEXT,
    channel TEXT,
    timestamp TIMESTAMP
);
"

psql -d crl_risk_lifecycle -c "\COPY accounts FROM 'accounts.csv' WITH (FORMAT csv, HEADER true);"
psql -d crl_risk_lifecycle -c "\COPY transactions FROM 'transactions.csv' WITH (FORMAT csv, HEADER true);"

echo "Database crl_risk_lifecycle created and loaded."
