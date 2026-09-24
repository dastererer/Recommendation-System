import pandas as pd
import numpy as np
import requests
import polars as pl
from dataclasses import dataclass
from typing import Self


json_data = None


# important endpoints output examples:
# GET /api/recsys/freelancers/id
# {
#   "success": true,
#   "data": {
#     "user_id": 7,
#     "first_name": "Demo",
#     "last_name": "Freelancer",
#     "display_name": "Demo Freelancer",
#     "headline": "Python developer",
#     "title": "Backend developer",
#     "description": "Develop APIs and data processing tools.",
#     "location": "Poland",
#     "hourly_rate": 30,
#     "experience_level": "intermediate",
#     "jobs_completed": 0,
#     "skills": ["Python", "SQL"],
#     "rating": {
#       "average": null,
#       "count": 0
#     }
#   }
# }
# GET /api/recsys/jobs
# {
#   "success": true,
#   "data": {
#     "jobs": [
#       {
#         "job_id": 42,
#         "source_job_id": "upwork-demo-001",
#         "is_imported": true,
#         "employer_id": 105,
#         "category_id": 1,
#         "category_name": "AI & Machine Learning",
#         "title": "Build a data processing tool",
#         "description": "Process datasets using Python and SQL.",
#         "budget": 600,
#         "job_type": "fixed",
#         "experience_level": "intermediate",
#         "status": "Open",
#         "duration_estimate": null,
#         "deadline": null,
#         "is_remote": true,
#         "location": null,
#         "skills": ["Python", "SQL"],
#         "applicants": {
#           "historical": 35,
#           "live": 3,
#           "total": 38
#         },
#         "employer": {
#           "source_employer_id": "job:upwork-demo-001",
#           "company_name": "Imported Client",
#           "location": "United States",
#           "rating": {
#             "historical_average": 4.8,
#             "historical_count": 25,
#             "live_average": 5,
#             "live_count": 2,
#             "effective_average": 4.814814814814815,
#             "total_count": 27
#           },
#           "spending": {
#             "historical": 10000,
#             "live": 500,
#             "total": 10500
#           }
#         }
#       }
#     ],
#     "total": 32182,
#     "limit": 500,
#     "offset": 0
#   }
# }

# user_id: int
# job_id: int
# semantic_score: float
# applicants_num: int
# client_country: str
# client_rating: float
# client_spending: float
# feedback_num: int
# job_skills: frozenset[str]
# freelancer_skills: frozenset[str]
# job_rate: float
# expected_rate: float
# job_experience: str
# freelancer_experience: str
# freelancer_country: str


df 




# userId = None
#
# endpoints = {
#     "jobs_import": "/api/recsys/import/jobs",
#     "jobs": "/api/recsys/jobs",
#     "freelancer_profile": f"/api/recsys/freelancers/:{userId}"
#
# }
#
#
# EXPERIENCE_COMPATIBILITY = {
#     "junior": {
#         "junior": 1.0,
#         "mid": 0.4,
#         "senior": 0.0,
#     },
#     "mid": {
#         "junior": 0.8,
#         "mid": 1.0,
#         "senior": 0.4,
#     },
#     "expert": {
#         "junior": 0.55,
#         "mid": 0.9,
#         "senior": 1.0,
#     },
# }
#
# def skills_score(job_skills: np.ndarray, freelancer_skills: np.ndarray):
#     intersection = np.intersect1d(job_skills, freelancer_skills)
#     return len(intersection) / len(job_skills) if len(job_skills) != 0 and len(freelancer_skills) != 0 else None
#
# def rate_score(job_rate: float, expected_rate: float, a: float = 1.5):
#     d = max(abs(expected_rate - job_rate)/expected_rate, 0)
#     return np.exp(-a*d) if job_rate is not None and expected_rate is not None else None
#
# def experience_score(job_experience: str, freelancer_experience: str):
#     return EXPERIENCE_COMPATIBILITY[
#         freelancer_experience
#     ][job_experience] if job_experience is not None and freelancer_experience is not None else None
#
# def competition_score(applicants_num: int, max_applicants_num: int):
#     return 1 - (np.log(1 + min(applicants_num, max_applicants_num)) / np.log(1 + max_applicants_num))
#
# def rating_score(job_rating: float, feedback_num: int, avg_rating: float, priority: float = 10):
#     bayes_score = (feedback_num * job_rating + priority * avg_rating) / (feedback_num + priority)
#     normalized_score = bayes_score / 5.0
#     return normalized_score if job_rating is not None else avg_rating
#
# def spend_score(spent_historical: float, spent_historical_95: float):
#     return np.clip(np.log(1 + spent_historical) / np.log(1 + spent_historical_95), 0, 1)
#
# def client_trust_score(x: float, y: float, job_rating: float, feedback_num: int, avg_rating: float, spent_historical: float, spent_historical_95: float, priority: float = 10):
#     return x * rating_score(job_rating, feedback_num, avg_rating, priority) + y * spend_score(spent_historical, spent_historical_95)
#
#
# def fit_score(x, y, z, w, semantic_match: float, job_skills: np.ndarray, freelancer_skills: np.ndarray, job_rate: float, expected_rate: float, job_experience: str, freelancer_experience: str, a: float = 1.5 ):
#     return x * semantic_match + y * skills_score(job_skills, freelancer_skills) + z * rating_score(job_rate, expected_rate, a) + w * experience_score(job_experience, freelancer_experience)
#
# def opportunity_score(x, y, job_rating: float, feedback_num: int, avg_rating: float, spent_historical: float, spent_historical_95: float, applicants_num: int, max_applicants_num: int, priority: float = 10):
#     return x * client_trust_score(0.7, 0.3, job_rating, feedback_num, avg_rating, spent_historical, spent_historical_95, priority) + y * competition_score(applicants_num, max_applicants_num)
#
# def final_score(fit_score, opportunity_score):
#     return np.clip(fit_score * (0.9 + 0.2 * opportunity_score), 0, 1)




