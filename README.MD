# Problem Statement
Design an Agentic workflow to train and predict a model for h1n1 vaccine usage. 

Data: https://github.com/emykes/Flu_Vaccination_ML/tree/main?tab=readme-ov-file

# Install
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
OPENAI_API_KEY=sk-....
python langgraph_agent.py

Example 1: Relearn from the data

Example 2: Will this is user adopt h1n1:
        "h1n1_concern": 1,
        "h1n1_knowledge": 1,
        "behavioral_face_mask": 0,
        "behavioral_large_gatherings": 0,
        "behavioral_wash_hands": 0,
        "behavioral_outside_home": 0,
        "behavioral_touch_face": 0,
        "child_under_6_months": 0,
        "health_worker": 0,
        "opinion_h1n1_vacc_effective": 0,
        "opinion_h1n1_risk": 0,
        "opinion_h1n1_sick_from_vacc": 0,
        "age_group": "18 - 34 Years",
        "education": "12 Years",
        "race": "White",
        "sex": "Male",
        "income_poverty": "Below Poverty",
        "marital_status": "Not Married",
        "rent_or_own": "Rent",
        "employment_status": "Employed",
        "census_msa": "Non-MSA",
        "hhs_geo_region": "xqwwgdyp",
        "behavioral_social_distance": 0,
        "employment_industry":"atmlpfrs",
        "employment_occupation":"hfxkjkmi"
