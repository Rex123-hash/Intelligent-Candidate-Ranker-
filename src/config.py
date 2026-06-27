"""Central configuration: scoring weights, keyword lexicons, paths, constants.
All tunable numbers live here so the methodology is readable and defensible."""
from datetime import date
from pathlib import Path

# --- paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "candidates.jsonl"
ARTIFACTS = ROOT / "artifacts"
EMB_FILE = ARTIFACTS / "embeddings.npy"
IDS_FILE = ARTIFACTS / "candidate_ids.npy"
JD_EMB_FILE = ARTIFACTS / "jd_embedding.npy"

# --- model ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# --- reference date for recency / tenure math ---
REFERENCE_DATE = date(2026, 6, 27)

# --- top-level score weights (semantic + features additive; penalties subtractive) ---
W_SEMANTIC = 0.45
W_FEATURES = 0.45          # features sub-score is a weighted mean in [0,1]
W_PENALTY = 0.30           # max total subtractive penalty

# --- feature sub-weights (sum ~1.0) ---
F_EXPERIENCE = 0.25
F_PRODUCT = 0.25
F_TITLE = 0.30
F_LOCATION = 0.12
F_EDUCATION = 0.08

# --- behavioral modifier floor ---
BEHAVIOR_FLOOR = 0.55      # worst-case multiplier

# --- experience target (years) ---
EXP_IDEAL = 7.0
EXP_HALF_WIDTH = 6.0       # triangular falloff width

# --- lexicons (all lowercase; matched as substrings) ---
SERVICES_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mindtree", "ltimindtree",
    "dxc", "mphasis", "hexaware", "igate", "syntel", "birlasoft", "coforge",
}
RELEVANT_TITLE_KEYWORDS = {
    "machine learning": 1.0, "ml engineer": 1.0, "ai engineer": 1.0,
    "data scientist": 0.9, "applied scientist": 1.0, "research engineer": 0.85,
    "nlp": 0.95, "search": 0.85, "retrieval": 0.95, "ranking": 0.95,
    "recommendation": 0.9, "software engineer": 0.6, "backend": 0.55,
    "data engineer": 0.55, "ai/ml": 1.0, "deep learning": 0.9,
}
NEGATIVE_TITLE_KEYWORDS = {
    "marketing", "sales", "operations manager", "mechanical", "civil",
    "accountant", "customer support", "graphic design", "hr manager",
    "recruiter", "content writer", "project manager", "business analyst",
}
AI_SKILL_KEYWORDS = {
    "machine learning", "deep learning", "nlp", "natural language",
    "embeddings", "retrieval", "ranking", "recommendation", "rag",
    "transformers", "pytorch", "tensorflow", "vector", "elasticsearch",
    "faiss", "information retrieval", "llm", "semantic search",
}
CV_SPEECH_KEYWORDS = {
    "computer vision", "image classification", "object detection",
    "speech recognition", "robotics", "opencv", "image segmentation",
}
LANGCHAIN_KEYWORDS = {"langchain", "llamaindex", "prompt engineering"}
TARGET_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "new delhi",
    "gurgaon", "gurugram", "bengaluru", "bangalore",
}
MANAGER_TITLE_KEYWORDS = {
    "manager", "director", "vp", "vice president", "head of", "architect", "lead",
}
ENGINEER_TITLE_KEYWORDS = {"engineer", "developer", "scientist", "programmer"}
