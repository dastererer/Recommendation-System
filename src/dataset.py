import json
from pathlib import Path
import html
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
upwork_garbage = {'looking', 'need', 'expert', 'project', 'work', 'term', 'hire', 'contractor'}
stop_words = list(stop_words.union(upwork_garbage))



PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "upwork-jobs.json"
INPUT_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "Final_Upwork_Dataset.csv"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "upwork-jobs.csv"
SKILLS_PATH = PROJECT_ROOT / "data" / "raw" / "skills.json"


with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = json.load(f)

with SKILLS_PATH.open("r", encoding="utf-8") as f:
    skills = json.load(f)
    skills = list(skills)
    skills_list_lower = [skill.lower() for skill in skills]


COLUMNS_TO_DROP = config["COLOUMS_TO_DROP"]
USE_STOPWORDS = config["USE_STOPWORDS"]


APPLICANTS_MAPPING = {
    'Less than 5': 3.0,
    '5 to 10': 7.5,
    '10 to 15': 12.5,
    '15 to 20': 17.5,
    '20 to 50': 35.0,
    '50+': 60.0
}

TIME_LIMITATION_MAPPING = {
    '3 to 6 months, Less than 30 hrs/week': 18 * 20,
    '3 to 6 months, 30+ hrs/week': 18 * 40,
    '3 to 6 months, Hours to be determined': 18 * 30,
    '3 to 6 months, not_sure': 18 * 30,

    'More than 6 months, Less than 30 hrs/week': 30 * 20,
    'More than 6 months, 30+ hrs/week': 30 * 40,
    'More than 6 months, Hours to be determined': 30 * 30,
    'More than 6 months, not_sure': 30 * 30,

    '1 to 3 months, Less than 30 hrs/week': 8 * 20,
    '1 to 3 months, 30+ hrs/week': 8 * 40,
    '1 to 3 months, Hours to be determined': 8 * 30,
    '1 to 3 months, not_sure': 8 * 30,

    'Less than 1 month, Less than 30 hrs/week': 3 * 20,
    'Less than 1 month, 30+ hrs/week': 3 * 40,
    'Less than 1 month, Hours to be determined': 3 * 30,
    'Less than 1 month, not_sure': 3 * 30
}

PAYMENT_MAPPING = {
    'Hourly': 1,
    'Fixed-price': 0,
}

CATEGORY_COLS = [f"Category_{i}" for i in range(1, 10)]


def drop_columns(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    existing_columns_to_drop = [
        column for column in columns_to_drop if column in df.columns
    ]

    return df.drop(existing_columns_to_drop, axis=1)


def clean_raw_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r'https?://\S+', '', text)

    text = html.unescape(text)
    text = text.replace("<null>", "")
    text = re.sub(r"<[^>]+>", "", text)

    text = text.replace("/", " ").replace("-", " ")
    text = re.sub(r"[.,!?—:;\"()\[\]{}*_\r\n]", " ", text)

    words = text.lower().split()

    if USE_STOPWORDS:
        words = [word for word in words if word not in stop_words]

    return " ".join(words).strip()


def calculate_hourly_rate(df: pd.DataFrame) -> pd.Series:
    start_clean = df["Start_rate"].astype(str).str.replace(r'[^\d.]', '', regex=True)
    end_clean = df["End_rate"].astype(str).str.replace(r'[^\d.]', '', regex=True)
    cost_clean = df["Job_Cost"].astype(str).str.replace(r'[^\d.]', '', regex=True)

    start_arr = pd.to_numeric(start_clean, errors="coerce")
    end_arr = pd.to_numeric(end_clean, errors="coerce")
    job_cost = pd.to_numeric(cost_clean, errors="coerce")
    job_cost = np.where(np.isnan(job_cost), 0.0, job_cost)

    time_limitation_clean = df["Time_Limitation"].str.strip()
    total_project_hours = time_limitation_clean.map(TIME_LIMITATION_MAPPING).fillna(120.0)

    start_is_nan = np.isnan(start_arr)
    end_is_nan = np.isnan(end_arr)

    cond_list = [
        ~start_is_nan & ~end_is_nan,
        start_is_nan & ~end_is_nan,
        ~start_is_nan & end_is_nan
    ]
    choice_list = [
        (start_arr + end_arr) / 2,
        end_arr / 2,
        start_arr
    ]

    hourly_avg = np.select(cond_list, choice_list, default=np.nan)

    fixed_calculated_rate = job_cost / total_project_hours

    is_valid_hourly = ~np.isnan(hourly_avg) & (hourly_avg > 4)

    hour_rate = np.where(
        is_valid_hourly,
        hourly_avg,
        fixed_calculated_rate
    )

    return pd.Series(hour_rate, index=df.index).fillna(0.0)


def process_upwork_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Row Filtering
    df = df.dropna(subset=["Description", "Job Title"])
    df = df.dropna(subset=CATEGORY_COLS, how="all")
    df = df[df["Description"].str.strip() != ""]
    df = df[df["Job Title"].str.strip() != ""]
    df = df.reset_index(drop=True)

    # Text Cleaning
    df["Job Title"] = df["Job Title"].apply(clean_raw_text)
    df["Description"] = df["Description"].apply(clean_raw_text)

    # Skill Extraction
    df["skills"] = df[CATEGORY_COLS].apply(
        lambda row: [str(item).strip().lower() for item in row if pd.notna(item) and str(item).strip() != ""],
        axis=1
    )

    df["EX_level_demand"] = df["EX_level_demand"].replace({"Expert": "senior", "Intermediate": "mid", "Beginner": "junior"})

    # Feature Mapping
    df["Applicants_Num"] = df["Applicants_Num"].map(APPLICANTS_MAPPING).fillna(-1)
    df["is_hourly"] = df["Payment_type"].map(PAYMENT_MAPPING).fillna(-1)

    # Rate Calculation
    df["hour_rate"] = calculate_hourly_rate(df)

    # Final Cleanup
    columns_to_drop = CATEGORY_COLS + ["Payment_type", "Time_Limitation", "Job_Cost", "Start_rate", "End_rate", "is_hourly"]
    df = df[df["hour_rate"] > 4.0].reset_index(drop=True)
    df = drop_columns(df, columns_to_drop)

    return df

if __name__ == "__main__":
    print(f"Starting preprocessing pipeline for {INPUT_CSV_PATH}...")

    df_raw = pd.read_csv(INPUT_CSV_PATH)
    print(f"Rows loaded: {df_raw.shape[0]}\nColumns loaded: {df_raw.shape[1]}")

    df_processed = process_upwork_dataset(df_raw)

    if COLUMNS_TO_DROP:
        df_processed = drop_columns(df_processed, COLUMNS_TO_DROP)

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_processed.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")

    print(f"Preprocessing finished. File saved to: {OUTPUT_CSV_PATH}")
    print(f"Rows total: {df_processed.shape[0]}\nColumns total: {df_processed.shape[1]}")