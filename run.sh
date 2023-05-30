PYTHONPATH=. python src/data/tcga/process_tcga.py
PYTHONPATH=. python src/data/simulation_data.py
PYTHONPATH=. python src/main_outcome.py
PYTHONPATH=. python src/main_density.py
PYTHONPATH=. python src/main_policy.py 
PYTHONPATH=. python src/main_policy.py ++policy.unadjusted=True
PYTHONPATH=. python src/main_outcome_MLP.py
PYTHONPATH=. python src/main_policy_MLP.py 
PYTHONPATH=. python src/main_policy_MLP.py ++policy.unadjusted=True


