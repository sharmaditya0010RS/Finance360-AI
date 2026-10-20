CREATE SCHEMA IF NOT EXISTS analytics;

CREATE OR REPLACE VIEW analytics.portfolio_overview AS
SELECT
    COUNT(*) AS application_count,
    COUNT(*) FILTER (WHERE target = 1) AS repayment_difficulty_count,
    ROUND(AVG(target::NUMERIC) * 100, 4) AS repayment_difficulty_rate_pct,
    ROUND(SUM(amt_credit)::NUMERIC, 2) AS total_requested_credit,
    ROUND(AVG(amt_credit)::NUMERIC, 2) AS average_requested_credit,
    ROUND(
        SUM(amt_credit) FILTER (WHERE target = 1)::NUMERIC,
        2
    ) AS difficulty_group_requested_credit,
    ROUND(
        100 * SUM(amt_credit) FILTER (WHERE target = 1)::NUMERIC
        / NULLIF(SUM(amt_credit)::NUMERIC, 0),
        4
    ) AS difficulty_group_credit_share_pct,
    COUNT(*) FILTER (
        WHERE financial_data_quality_flag = 1
    ) AS flagged_application_count
FROM staging.credit_applications;

CREATE OR REPLACE VIEW analytics.income_segments AS
WITH segmented AS (
    SELECT
        CASE
            WHEN amt_income_total < 90000 THEN '01 - Below 90k'
            WHEN amt_income_total < 180000 THEN '02 - 90k to 180k'
            WHEN amt_income_total < 360000 THEN '03 - 180k to 360k'
            ELSE '04 - 360k and above'
        END AS income_segment,
        target,
        amt_credit,
        credit_to_income_ratio,
        annuity_to_income_ratio
    FROM staging.credit_applications
    WHERE amt_income_total > 0
)
SELECT
    income_segment,
    COUNT(*) AS application_count,
    COUNT(*) FILTER (WHERE target = 1) AS repayment_difficulty_count,
    ROUND(AVG(target::NUMERIC) * 100, 4) AS repayment_difficulty_rate_pct,
    ROUND(SUM(amt_credit)::NUMERIC, 2) AS requested_credit,
    ROUND(AVG(credit_to_income_ratio)::NUMERIC, 4) AS avg_credit_to_income,
    ROUND(AVG(annuity_to_income_ratio)::NUMERIC, 4) AS avg_annuity_to_income
FROM segmented
GROUP BY income_segment;

CREATE OR REPLACE VIEW analytics.contract_type_risk AS
SELECT
    name_contract_type,
    COUNT(*) AS application_count,
    COUNT(*) FILTER (WHERE target = 1) AS repayment_difficulty_count,
    ROUND(AVG(target::NUMERIC) * 100, 4) AS repayment_difficulty_rate_pct,
    ROUND(SUM(amt_credit)::NUMERIC, 2) AS requested_credit,
    ROUND(AVG(amt_credit)::NUMERIC, 2) AS average_requested_credit
FROM staging.credit_applications
GROUP BY name_contract_type;