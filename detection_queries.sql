-- Customer Risk Lifecycle: structuring & mule-network detection
-- PostgreSQL 16. Run against the accounts / transactions tables.

-- ============================================================
-- 1. Rolling 7-day inflow frequency + sum per account
--    Catches high-frequency, low-value structuring patterns
--    invisible at the single-transaction level.
-- ============================================================
CREATE OR REPLACE VIEW rolling_inflow_velocity AS
SELECT
    account_id,
    timestamp,
    amount,
    COUNT(*) OVER (
        PARTITION BY account_id
        ORDER BY timestamp
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
    ) AS inflow_count_7d,
    SUM(amount) OVER (
        PARTITION BY account_id
        ORDER BY timestamp
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
    ) AS inflow_sum_7d,
    AVG(amount) OVER (
        PARTITION BY account_id
        ORDER BY timestamp
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
    ) AS avg_inflow_amount_7d
FROM transactions
WHERE direction = 'inflow';

-- Structuring flag: many small transactions, high frequency, low individual amount
CREATE OR REPLACE VIEW structuring_candidates AS
SELECT
    account_id,
    MAX(inflow_count_7d) AS max_7d_inflow_count,
    MAX(inflow_sum_7d) AS max_7d_inflow_sum,
    AVG(avg_inflow_amount_7d) AS avg_amount
FROM rolling_inflow_velocity
GROUP BY account_id
HAVING MAX(inflow_count_7d) >= 10          -- high frequency
   AND AVG(avg_inflow_amount_7d) < 700      -- low individual value (below the €1,000 GwG transfer-information threshold)
ORDER BY max_7d_inflow_count DESC;


-- ============================================================
-- 2. Time-to-outflow per account
--    Catches rapid pass-through / funnel-account behaviour:
--    funds moved out almost immediately after arriving.
-- ============================================================
CREATE OR REPLACE VIEW inflow_outflow_pairs AS
SELECT
    i.account_id,
    i.transaction_id AS inflow_tx,
    i.amount AS inflow_amount,
    i.timestamp AS inflow_time,
    o.transaction_id AS outflow_tx,
    o.timestamp AS outflow_time,
    EXTRACT(EPOCH FROM (o.timestamp - i.timestamp)) / 3600.0 AS hours_to_outflow
FROM transactions i
JOIN transactions o
  ON i.account_id = o.account_id
 AND o.direction = 'outflow'
 AND o.timestamp > i.timestamp
 AND o.timestamp <= i.timestamp + INTERVAL '24 hours'
WHERE i.direction = 'inflow';

CREATE OR REPLACE VIEW rapid_pass_through_accounts AS
SELECT
    account_id,
    COUNT(*) AS fast_pairs,
    AVG(hours_to_outflow) AS avg_hours_to_outflow
FROM inflow_outflow_pairs
GROUP BY account_id
HAVING AVG(hours_to_outflow) < 12   -- funds typically leave within half a day
ORDER BY fast_pairs DESC;


-- ============================================================
-- 3. Network link detection: two independent signals
--    (a) shared counterparties across accounts
--    (b) shared onboarding device, the realistic German/EU signal:
--        remote video-ident onboarding (IDnow/WebID-style) completed
--        from the same device across multiple "unrelated" identities,
--        a well-documented mule-recruitment indicator.
-- ============================================================
CREATE OR REPLACE VIEW shared_counterparty_links AS
SELECT
    t1.account_id AS account_a,
    t2.account_id AS account_b,
    t1.counterparty_id,
    COUNT(*) AS shared_transactions
FROM transactions t1
JOIN transactions t2
  ON t1.counterparty_id = t2.counterparty_id
 AND t1.account_id < t2.account_id   -- avoid duplicate pairs and self-joins
GROUP BY t1.account_id, t2.account_id, t1.counterparty_id
HAVING COUNT(*) >= 2
ORDER BY shared_transactions DESC;

CREATE OR REPLACE VIEW shared_device_links AS
SELECT
    a1.account_id AS account_a,
    a2.account_id AS account_b,
    a1.onboarding_device_id
FROM accounts a1
JOIN accounts a2
  ON a1.onboarding_device_id = a2.onboarding_device_id
 AND a1.account_id < a2.account_id;


-- ============================================================
-- 4. Combined structuring-likelihood score
--    Brings velocity + pass-through + network evidence together
--    into a single ranked output, the analyst's starting point.
-- ============================================================
CREATE OR REPLACE VIEW structuring_likelihood_score AS
SELECT
    a.account_id,
    a.initial_risk_category,
    COALESCE(s.max_7d_inflow_count, 0) AS velocity_signal,
    COALESCE(r.fast_pairs, 0) AS pass_through_signal,
    COALESCE(n.counterparty_link_count, 0) AS counterparty_network_signal,
    COALESCE(d.device_link_count, 0) AS device_network_signal,
    (COALESCE(s.max_7d_inflow_count, 0) * 1.0
     + COALESCE(r.fast_pairs, 0) * 2.0
     + COALESCE(n.counterparty_link_count, 0) * 3.0
     + COALESCE(d.device_link_count, 0) * 3.0) AS structuring_score
FROM accounts a
LEFT JOIN structuring_candidates s ON a.account_id = s.account_id
LEFT JOIN rapid_pass_through_accounts r ON a.account_id = r.account_id
LEFT JOIN (
    SELECT account_id, COUNT(*) AS counterparty_link_count
    FROM (
        SELECT account_a AS account_id FROM shared_counterparty_links
        UNION ALL
        SELECT account_b AS account_id FROM shared_counterparty_links
    ) x
    GROUP BY account_id
) n ON a.account_id = n.account_id
LEFT JOIN (
    SELECT account_id, COUNT(*) AS device_link_count
    FROM (
        SELECT account_a AS account_id FROM shared_device_links
        UNION ALL
        SELECT account_b AS account_id FROM shared_device_links
    ) y
    GROUP BY account_id
) d ON a.account_id = d.account_id
WHERE COALESCE(s.max_7d_inflow_count, 0) > 0
   OR COALESCE(r.fast_pairs, 0) > 0
ORDER BY structuring_score DESC
LIMIT 30;
