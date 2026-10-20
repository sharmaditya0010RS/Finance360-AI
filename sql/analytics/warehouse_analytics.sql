CREATE OR REPLACE VIEW analytics.warehouse_portfolio_summary AS
SELECT
    COUNT(*) AS application_count,
    COUNT(*) FILTER (WHERE f.target = 1) AS repayment_difficulty_count,
    ROUND(AVG(f.target::NUMERIC) * 100, 4) AS repayment_difficulty_rate_pct,
    ROUND(SUM(f.amt_credit)::NUMERIC, 2) AS total_requested_credit,
    ROUND(AVG(f.amt_credit)::NUMERIC, 2) AS average_requested_credit,
    ROUND(SUM(f.amt_income_total)::NUMERIC, 2) AS total_reported_income,
    COUNT(*) FILTER (
        WHERE f.employment_sentinel_flag = 1
    ) AS employment_sentinel_count,
    COUNT(*) FILTER (
        WHERE f.financial_data_quality_flag = 1
    ) AS financial_quality_flag_count
FROM warehouse.fact_credit_application AS f;

CREATE OR REPLACE VIEW analytics.warehouse_contract_performance AS
SELECT
    d.contract_type_key,
    d.contract_type_name,
    COUNT(*) AS application_count,
    COUNT(*) FILTER (WHERE f.target = 1) AS repayment_difficulty_count,
    ROUND(AVG(f.target::NUMERIC) * 100, 4) AS repayment_difficulty_rate_pct,
    ROUND(SUM(f.amt_credit)::NUMERIC, 2) AS total_requested_credit,
    ROUND(AVG(f.amt_credit)::NUMERIC, 2) AS average_requested_credit,
    ROUND(AVG(f.credit_to_income_ratio)::NUMERIC, 4) AS avg_credit_to_income_ratio
FROM warehouse.fact_credit_application AS f
JOIN warehouse.dim_contract_type AS d
    ON f.contract_type_key = d.contract_type_key
GROUP BY
    d.contract_type_key,
    d.contract_type_name;

CREATE OR REPLACE VIEW analytics.warehouse_income_type_performance AS
SELECT
    d.income_type_key,
    d.income_type_name,
    COUNT(*) AS application_count,
    COUNT(*) FILTER (WHERE f.target = 1) AS repayment_difficulty_count,
    ROUND(AVG(f.target::NUMERIC) * 100, 4) AS repayment_difficulty_rate_pct,
    ROUND(SUM(f.amt_credit)::NUMERIC, 2) AS total_requested_credit,
    ROUND(AVG(f.amt_credit)::NUMERIC, 2) AS average_requested_credit,
    ROUND(AVG(f.annuity_to_income_ratio)::NUMERIC, 4) AS avg_annuity_to_income_ratio
FROM warehouse.fact_credit_application AS f
JOIN warehouse.dim_income_type AS d
    ON f.income_type_key = d.income_type_key
GROUP BY
    d.income_type_key,
    d.income_type_name;