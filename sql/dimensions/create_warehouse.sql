CREATE SCHEMA IF NOT EXISTS warehouse;

CREATE TABLE IF NOT EXISTS warehouse.dim_contract_type (
    contract_type_key INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    contract_type_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS warehouse.dim_income_type (
    income_type_key INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    income_type_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS warehouse.dim_education (
    education_key INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    education_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS warehouse.dim_housing (
    housing_key INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    housing_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS warehouse.dim_family_status (
    family_status_key INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    family_status_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS warehouse.fact_credit_application (
    sk_id_curr BIGINT PRIMARY KEY,
    contract_type_key INTEGER NOT NULL
        REFERENCES warehouse.dim_contract_type(contract_type_key),
    income_type_key INTEGER NOT NULL
        REFERENCES warehouse.dim_income_type(income_type_key),
    education_key INTEGER NOT NULL
        REFERENCES warehouse.dim_education(education_key),
    housing_key INTEGER NOT NULL
        REFERENCES warehouse.dim_housing(housing_key),
    family_status_key INTEGER NOT NULL
        REFERENCES warehouse.dim_family_status(family_status_key),
    target SMALLINT NOT NULL CHECK (target IN (0, 1)),
    amt_income_total DOUBLE PRECISION,
    amt_credit DOUBLE PRECISION,
    amt_annuity DOUBLE PRECISION,
    amt_goods_price DOUBLE PRECISION,
    days_birth INTEGER,
    days_employed INTEGER,
    cnt_children INTEGER,
    cnt_fam_members DOUBLE PRECISION,
    flag_own_car TEXT,
    flag_own_realty TEXT,
    ext_source_1 DOUBLE PRECISION,
    ext_source_2 DOUBLE PRECISION,
    ext_source_3 DOUBLE PRECISION,
    employment_sentinel_flag SMALLINT,
    age_years DOUBLE PRECISION,
    employment_years DOUBLE PRECISION,
    age_invalid_flag SMALLINT,
    employment_invalid_flag SMALLINT,
    income_invalid_flag SMALLINT,
    credit_invalid_flag SMALLINT,
    annuity_invalid_flag SMALLINT,
    credit_to_income_ratio DOUBLE PRECISION,
    annuity_to_income_ratio DOUBLE PRECISION,
    credit_to_goods_ratio DOUBLE PRECISION,
    income_per_family_member DOUBLE PRECISION,
    ext_source_available_count SMALLINT,
    ext_source_mean DOUBLE PRECISION,
    ext_source_invalid_flag SMALLINT,
    financial_data_quality_flag SMALLINT
);

CREATE INDEX IF NOT EXISTS idx_fact_credit_contract_type
    ON warehouse.fact_credit_application(contract_type_key);

CREATE INDEX IF NOT EXISTS idx_fact_credit_income_type
    ON warehouse.fact_credit_application(income_type_key);

CREATE INDEX IF NOT EXISTS idx_fact_credit_education
    ON warehouse.fact_credit_application(education_key);

CREATE INDEX IF NOT EXISTS idx_fact_credit_housing
    ON warehouse.fact_credit_application(housing_key);

CREATE INDEX IF NOT EXISTS idx_fact_credit_family_status
    ON warehouse.fact_credit_application(family_status_key);