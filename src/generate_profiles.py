# import json
#
# import faiss
# import ast
# import random
# import numpy as np
import pandas as pd
import requests
from faker import Faker
# import seaborn as sns
from pathlib import Path
# import matplotlib.pyplot as plt
# from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path.cwd().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
INPUT_CSV_PATH = ARTIFACTS_DIR / "vacancies_indexed.csv"

df = pd.read_csv(INPUT_CSV_PATH)

def generate_client_data(total_needed: int):
    fake = Faker()
    df["client_first_name"] = [fake.first_name() for _ in range(total_needed)]
    df["client_last_name"] = [fake.last_name() for _ in range(total_needed)]
    df["client_email"] = [fake.email() for _ in range(total_needed)]
    df["client_company_name"] = [fake.company_name_word() for _ in range(total_needed)]
    df.to_csv(INPUT_CSV_PATH, index=False)



