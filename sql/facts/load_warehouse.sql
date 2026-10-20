TRUNCATE TABLE
    warehouse.fact_credit_application,
    warehouse.dim_contract_type,
    warehouse.dim_income_type,
    warehouse.dim_education,
    warehouse.dim_housing,
    warehouse.dim_family_status
RESTART IDENTITY;

INSERT INTO warehouse.dim_contract_type (contract_type_name)
SELECT DISTINCT name_contract_type
FROM staging.credit_applications
ORDER BY name_contract_type;

INSERT INTO warehouse.dim_income_type (income_type_name)
SELECT DISTINCT name_income_type
FROM staging.credit_applications
ORDER BY name_income_type;

INSERT INTO warehouse.dim_education (education_name)
SELECT DISTINCT name_education_type
FROM staging.credit_applications
ORDER BY name_education_type;

INSERT INTO warehouse.dim_housing (housing_name)
SELECT DISTINCT name_housing_type
FROM staging.credit_applications
ORDER BY name_housing_type;

INSERT INTO warehouse.dim_family_status (family_status_name)
SELECT DISTINCT name_family_status
FROM staging.credit_applications
ORDER BY name_family_status;

INSERT INTO warehouse.fact_credit_application (
    sk_id_curr,
    contract_type_key,
    income_type_key,
    education_key,
    housing_key,
    family_status_key,
    target,
    amt_income_total,
    amt_credit,
    amt_annuity,
    amt_goods_price,
    days_birth,
    days_employed,
    cnt_children,
    cnt_fam_members,
    flag_own_car,
    flag_own_realty,
    ext_source_1,
    ext_source_2,
    ext_source_3,
    employment_sentinel_flag,
    age_years,
    employment_years,
    age_invalid_flag,
    employment_invalid_flag,
    income_invalid_flag,
    credit_invalid_flag,
    annuity_invalid_flag,
    credit_to_income_ratio,
    annuity_to_income_ratio,
    credit_to_goods_ratio,
    income_per_family_member,
    ext_source_available_count,
    ext_source_mean,
    ext_source_invalid_flag,
    financial_data_quality_flag
)
SELECT
    s.sk_id_curr,
    c.contract_type_key,
    i.income_type_key,
    e.education_key,
    h.housing_key,
    f.family_status_key,
    s.target,
    s.amt_income_total,
    s.amt_credit,
    s.amt_annuity,
    s.amt_goods_price,
    s.days_birth,
    s.days_employed,
    s.cnt_children,
    s.cnt_fam_members,
    s.flag_own_car,
    s.flag_own_realty,
    s.ext_source_1,
    s.ext_source_2,
    s.ext_source_3,
    s.employment_sentinel_flag,
    s.age_years,
    s.employment_years,
    s.age_invalid_flag,
    s.employment_invalid_flag,
    s.income_invalid_flag,
    s.credit_invalid_flag,
    s.annuity_invalid_flag,
    s.credit_to_income_ratio,
    s.annuity_to_income_ratio,
    s.credit_to_goods_ratio,
    s.income_per_family_member,
    s.ext_source_available_count,
    s.ext_source_mean,
    s.ext_source_invalid_flag,
    s.financial_data_quality_flag
FROM staging.credit_applications AS s
JOIN warehouse.dim_contract_type AS c
    ON s.name_contract_type = c.contract_type_name
JOIN warehouse.dim_income_type AS i
    ON s.name_income_type = i.income_type_name
JOIN warehouse.dim_education AS e
    ON s.name_education_type = e.education_name
JOIN warehouse.dim_housing AS h
    ON s.name_housing_type = h.housing_name
JOIN warehouse.dim_family_status AS f
    ON s.name_family_status = f.family_status_name;