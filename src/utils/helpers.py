import os
import json
import time
from typing import Dict,Any,List,Optional
import yaml

DEFAULT_CONFIG={
    "app":{
        "name":"OmniDoc AI",
        "description":"Academic RAG Assistant for Engineering Students",
        "version":"2.0.0"
    },
    "database":{
        "persist_directory":"./pdf_db/chromadb",
        "collection_name":"Document__C",
        "cloud_zip_path":"./PDF_db.zip"
    },
    "embeddings":{
        "model_name":"all-MiniLM-L6-v2",
        "device":"cpu"
    },
    "chunking":{
        "chunk_size":1000,
        "chunk_overlap":200
    },
    "retrieval":{
        "top_k":8,
        "min_context_length":60,
        "keyword_min_length":4,
        "relevance_threshold":0.25
    },
    "llm":{
        "default_groq_model":"openai/gpt-oss-120b",
        "fast_groq_model":"openai/gpt-oss-20b",
        "mini_groq_model":"groq/compound-mini",
        "default_ollama_model":"llama3.2:latest",
        "temperature":0.0
    },
    "paths": {
        "data_dir":"./PDF_Data",
        "chat_sessions_dir":"./chat_history_sessions",
        "logs_dir":"./logs"
    }
}

def normalize_subject_name(subject: Optional[str]) -> Optional[str]:
    """Return the canonical database subject key regardless of case, title or alias."""
    if subject is None:
        return None

    raw=subject.strip()
    if not raw:
        return raw
    if raw.lower() in ("all subjects","all","global"):
        return "All Subjects"

    for canonical in SUBJECT_METADATA.keys():
        if canonical.lower()==raw.lower():
            return canonical

    for key,meta in SUBJECT_METADATA.items():
        title=meta.get("title","")
        if title.lower()==raw.lower():
            return key
        if title.lower().replace("&","and")==raw.lower().replace("&","and"):
            return key

    for abbrev,canonical in SUBJECT_ABBREV.items():
        if abbrev.lower()==raw.lower():
            return canonical

    return raw

def load_app_config(config_path: str="./config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with fallback to defaults."""
    if os.path.exists(config_path):
        try:
            with open(config_path,"r",encoding="utf-8") as f:
                cfg=yaml.safe_load(f)
                if isinstance(cfg,dict):
                    return cfg
        except Exception as e:
            print(f"Warning: Failed to parse config.yaml ({e}). Using defaults.")
    return DEFAULT_CONFIG


def get_groq_api_key(streamlit_context: bool=True) -> str:
    """Fetch Groq API key from Streamlit secrets or environment variables."""
    if streamlit_context:
        try:
            import streamlit as st
            if "GROQ_API_KEY" in st.secrets:
                return st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    return os.getenv("GROQ_API_KEY","").strip()


SUBJECT_METADATA = {
    "AI": {"title": "Artificial Intelligence", "category": "AI & Data Science", "icon": "🤖", "type": "Notes"},
    "AI-NLP Lab": {"title": "AI & NLP Lab", "category": "AI & Data Science", "icon": "🧪", "type": "Lab"},
    "ML notes": {"title": "Machine Learning", "category": "AI & Data Science", "icon": "🧠", "type": "Notes"},
    "ML Lab": {"title": "Machine Learning Lab", "category": "AI & Data Science", "icon": "🔬", "type": "Lab"},
    "NLP": {"title": "Natural Language Processing", "category": "AI & Data Science", "icon": "💬", "type": "Notes"},
    "Neural network and deep learning": {"title": "Neural Networks & Deep Learning", "category": "AI & Data Science", "icon": "🕸️", "type": "Notes"},
    "Reinforcement Learning": {"title": "Reinforcement Learning", "category": "AI & Data Science", "icon": "🎮", "type": "Notes"},
    "CN Notes": {"title": "Computer Networks", "category": "Networks & Security", "icon": "📡", "type": "Notes"},
    "CN Lab": {"title": "Computer Networks Lab", "category": "Networks & Security", "icon": "🔌", "type": "Lab"},
    "CNS": {"title": "Cryptography & Network Security", "category": "Networks & Security", "icon": "🔐", "type": "Notes"},
    "CNS Lab": {"title": "CNS Lab", "category": "Networks & Security", "icon": "🛡️", "type": "Lab"},
    "Cloud Computing": {"title": "Cloud Computing", "category": "Networks & Security", "icon": "☁️", "type": "Notes"},
    "SNA": {"title": "Social Network Analysis", "category": "Networks & Security", "icon": "🌐", "type": "Notes"},
    "CD Notes": {"title": "Compiler Design", "category": "Core CS & Systems", "icon": "⚙️", "type": "Notes"},
    "DAA Notes": {"title": "Design & Analysis of Algorithms", "category": "Core CS & Systems", "icon": "🧮", "type": "Notes"},
    "DBMS": {"title": "Database Management Systems", "category": "Core CS & Systems", "icon": "🗄️", "type": "Notes"},
    "DBMS Lab": {"title": "DBMS Lab", "category": "Core CS & Systems", "icon": "💾", "type": "Lab"},
    "OS": {"title": "Operating Systems", "category": "Core CS & Systems", "icon": "💻", "type": "Notes"},
    "COA": {"title": "Computer Organization & Architecture", "category": "Core CS & Systems", "icon": "🏗️", "type": "Notes"},
    "FLAT": {"title": "Formal Languages & Automata Theory", "category": "Core CS & Systems", "icon": "🔢", "type": "Notes"},
    "Software Engineering": {"title": "Software Engineering", "category": "Core CS & Systems", "icon": "🛠️", "type": "Notes"},
    "Devops": {"title": "DevOps", "category": "Core CS & Systems", "icon": "🚀", "type": "Notes"},
    "Devops lab": {"title": "DevOps Lab", "category": "Core CS & Systems", "icon": "🐳", "type": "Lab"},
    "Data structure": {"title": "Data Structures", "category": "Core CS & Systems", "icon": "🌳", "type": "Notes"},
    "BEFA": {"title": "Business Economics & Financial Analysis", "category": "Management & Electives", "icon": "📊", "type": "Notes"},
    "DM": {"title": "Discrete Mathematics / Data Mining", "category": "Management & Electives", "icon": "📐", "type": "Notes"},
    "DPPM": {"title": "Data Preparation & Pattern Mining", "category": "Management & Electives", "icon": "⛏️", "type": "Notes"},
    "Java": {"title": "Java Programming", "category": "Management & Electives", "icon": "☕", "type": "Notes"},
    "Java Lab": {"title": "Java Lab", "category": "Management & Electives", "icon": "🍵", "type": "Lab"},
    "MSF": {"title": "Management Science & Finance", "category": "Management & Electives", "icon": "📈", "type": "Notes"},
    "Organizational Behaviour": {"title": "Organizational Behaviour", "category": "Management & Electives", "icon": "🏢", "type": "Notes"},
    "POE": {"title": "Principles of Economics", "category": "Management & Electives", "icon": "💰", "type": "Notes"},
    "PP": {"title": "Python Programming", "category": "Management & Electives", "icon": "🐍", "type": "Notes"},
    "STM Notes": {"title": "Software Testing Methodologies", "category": "Management & Electives", "icon": "🧪", "type": "Notes"},
    "Semantic Web": {"title": "Semantic Web", "category": "Management & Electives", "icon": "🕸️", "type": "Notes"},
    "Total Quality Management": {"title": "Total Quality Management", "category": "Management & Electives", "icon": "🎯", "type": "Notes"},
    "WP Notes": {"title": "Web Programming", "category": "Management & Electives", "icon": "🖥️", "type": "Notes"},
    "ACS Lab": {"title": "Advanced Communication Systems Lab", "category": "Management & Electives", "icon": "📡", "type": "Lab"},
}

SIDEBAR_CATEGORIES = {
    "🤖 AI & Data Science": [
        "AI", "AI-NLP Lab", "ML notes", "ML Lab",
        "NLP", "Neural network and deep learning", "Reinforcement Learning"
    ],
    "📡 Networks & Security": [
        "CN Notes", "CN Lab", "CNS", "CNS Lab",
        "Cloud Computing", "SNA"
    ],
    "💻 Core CS & Systems": [
        "CD Notes", "DAA Notes", "DBMS", "DBMS Lab",
        "OS", "COA", "FLAT", "Software Engineering",
        "Devops", "Devops lab", "Data structure"
    ],
    "📊 Management & Electives": [
        "BEFA", "DM", "DPPM", "Java", "Java Lab",
        "MSF", "Organizational Behaviour", "POE",
        "PP", "STM Notes", "Semantic Web",
        "Total Quality Management", "WP Notes", "ACS Lab"
    ]
}

SUBJECT_SAMPLE_QUESTIONS = {
    "AI": ["Give AI lab list of experiments", "What is Turing Test?", "Explain A* search algorithm", "What is Heuristic Search?"],
    "AI-NLP Lab": ["List all NLP lab experiments", "What is tokenization in NLP?", "Explain Named Entity Recognition", "Give AI NLP lab programs list"],
    "ML notes": ["How many types of Machine Learning?", "Explain supervised vs unsupervised", "What is overfitting in ML?", "Explain Decision Tree algorithm"],
    "ML Lab": ["List all ML lab experiments", "How to implement KNN algorithm?", "Explain SVM with example", "What is Naive Bayes classifier?"],
    "NLP": ["What is NLP? Define it", "Explain parsing in NLP", "What is stemming and lemmatization?", "NLP applications in real life"],
    "Neural network and deep learning": ["What is Deep Learning?", "Explain Backpropagation algorithm", "What is CNN vs RNN?", "Define activation functions"],
    "Reinforcement Learning": ["What is RL? Define it", "Explain Q-learning algorithm", "What is Markov Decision Process?", "RL vs supervised learning"],
    "CN Notes": ["What is Computer Networks (CN)?", "Explain OSI model 7 layers", "What is TCP vs UDP?", "Explain IP addressing and subnetting"],
    "CN Lab": ["List all CN lab experiments", "What is socket programming?", "Explain ping and traceroute", "TCP vs UDP lab experiment"],
    "CNS": ["What is Cryptography?", "Explain RSA algorithm", "What is Digital Signature?", "Explain AES encryption"],
    "CNS Lab": ["List all CNS lab experiments", "Implement Caesar cipher", "What is DES algorithm?", "CNS lab programs list"],
    "Cloud Computing": ["What is Cloud Computing?", "Explain SaaS PaaS IaaS", "What is virtualization?", "Types of cloud deployment models"],
    "SNA": ["What is Social Network Analysis?", "Explain centrality measures", "What is graph theory in SNA?", "Network clustering algorithms"],
    "CD Notes": ["What are phases of Compiler Design?", "Explain lexical analysis", "What is syntax analysis / parsing?", "Explain code optimization"],
    "DAA Notes": ["What is DAA? Define DA", "Explain time complexity Big O", "What is Dynamic Programming?", "Explain Greedy algorithms"],
    "DBMS": ["What is DBMS?", "Explain normalization forms", "What is SQL vs NoSQL?", "Explain ER diagram"],
    "DBMS Lab": ["List all DBMS lab experiments", "Write SQL queries for joins", "Create database with DBMS lab", "Explain triggers and procedures"],
    "OS": ["What is Operating System?", "Explain process scheduling algorithms", "What is deadlock and its prevention?", "Explain memory management"],
    "COA": ["What is Computer Organization?", "Explain CPU architecture", "What is pipelining in COA?", "Explain memory hierarchy"],
    "FLAT": ["What is Automata Theory?", "Explain DFA vs NFA", "What is pushdown automaton?", "Explain Turing Machine"],
    "Software Engineering": ["What is SDLC?", "Explain Agile methodology", "What is software testing?", "Explain UML diagrams"],
    "Devops": ["What is DevOps?", "Explain CI/CD pipeline", "What is Docker and Kubernetes?", "DevOps tools overview"],
    "Devops lab": ["List DevOps lab experiments", "Setup Docker container", "Implement Jenkins CI/CD", "Git workflow in DevOps"],
    "Data structure": ["What are linear data structures?", "Explain trees and graphs", "What is sorting algorithms?", "Explain stack and queue"],
    "BEFA": ["What is business economics?", "Explain demand and supply", "Financial ratio analysis", "What is break-even analysis?"],
    "DM": ["What is Data Mining?", "Explain association rules", "What is clustering in DM?", "Data mining algorithms"],
    "DPPM": ["What is DPPM?", "Explain data preprocessing", "What is pattern mining?", "Feature engineering techniques"],
    "Java": ["What is OOP in Java?", "Explain Java inheritance", "What is Exception Handling in Java?", "Java collections framework"],
    "Java Lab": ["List all Java lab programs", "Implement Java thread program", "Java file handling program", "Write Java socket program"],
    "MSF": ["What is Management Science?", "Explain linear programming", "What is operations research?", "Financial management basics"],
    "Organizational Behaviour": ["What is OB?", "Explain motivation theories", "What is organizational culture?", "Leadership styles in OB"],
    "POE": ["What is Economics?", "Explain microeconomics vs macroeconomics", "What is GDP?", "Types of market structures"],
    "PP": ["What is Python?", "Python data types and variables", "Explain list comprehension", "Python OOP concepts"],
    "STM Notes": ["What is software testing?", "Black box vs white box testing", "What is unit testing?", "Explain test cases and test plans"],
    "Semantic Web": ["What is Semantic Web?", "Explain RDF and OWL", "What is ontology?", "SPARQL query language"],
    "Total Quality Management": ["What is TQM?", "Explain Six Sigma", "What is ISO standards?", "TQM tools and techniques"],
    "WP Notes": ["What is HTML and CSS?", "Explain JavaScript basics", "What is responsive web design?", "Web frameworks overview"],
    "ACS Lab": ["What is ACS Lab?", "List all ACS lab experiments", "ACS lab record notes overview", "Communication systems basics"],
    "All Subjects": ["What is Computer Networks?", "Give AI lab list of experiments", "Explain normalization in DBMS", "What are types of Machine Learning?"]
}

SUBJECT_SCOPE_CONTEXT = {
    "MSF": (
        "Subject scope: Management Science and Finance. Management science applies "
        "quantitative methods, models, and analytical decision-making techniques to "
        "business and organizational problems. The subject commonly includes operations "
        "research, statistics, optimization, and financial management."
    ),
    "POE": (
        "Subject scope: Principles of Economics. Economics studies how individuals, "
        "businesses, and governments allocate scarce resources. Core topics include "
        "demand, supply, markets, production, and macroeconomic indicators."
    ),
}

SUBJECT_ABBREV = {
    "STM Notes": {
        "cfg": "Control Flow Graph", "cn": "Control Flow Graph (CFG)",
        "dfg": "Data Flow Graph", "dff": "Data Flow",
        "jf": "Junction / Decision Flow (Domain Testing)",
        "tf": "Transaction Flowgraph", "bb": "Basic Block",
        "nra": "Node Reduction Algorithm", "mcdc": "Modified Condition / Decision Coverage",
        "dd": "Definition-Definition anomaly", "du": "Definition-Use",
        "ku": "Kill-Use", "dk": "Definition-Kill",
        "lcsaj": "Linear Code Sequence And Jump",
    },
    "ML notes": {
        "da": "Data Analysis", "ml": "Machine Learning",
        "svm": "Support Vector Machine", "knn": "K-Nearest Neighbors",
        "dt": "Decision Tree", "rf": "Random Forest",
        "nn": "Neural Network", "bp": "Backpropagation",
        "lr": "Linear Regression", "log": "Logistic Regression",
        "em": "Expectation Maximization", "pca": "Principal Component Analysis",
        "nb": "Naive Bayes", "adaboost": "Adaptive Boosting",
    },
    "ML Lab": {
        "svm": "Support Vector Machine", "knn": "K-Nearest Neighbors",
        "lr": "Linear Regression", "nb": "Naive Bayes",
    },
    "AI": {
        "bfs": "Breadth First Search", "dfs": "Depth First Search",
        "ids": "Iterative Deepening Search", "a*": "A Star Search",
        "csp": "Constraint Satisfaction Problem", "kb": "Knowledge Base",
        "pl": "Propositional Logic", "fol": "First Order Logic",
    },
    "Neural network and deep learning": {
        "ann": "Artificial Neural Network", "rnn": "Recurrent Neural Network",
        "cnn": "Convolutional Neural Network", "dnn": "Deep Neural Network",
        "lstm": "Long Short-Term Memory", "gru": "Gated Recurrent Unit",
        "gan": "Generative Adversarial Network", "ae": "Autoencoder",
        "nnd": "Neural Networks and Deep Learning", "nndl": "Neural Networks and Deep Learning",
        "bp": "Backpropagation", "rbf": "Radial Basis Function",
    },
    "CN Notes": {
        "cn": "Computer Networks", "osi": "Open Systems Interconnection model",
        "tcp": "Transmission Control Protocol", "udp": "User Datagram Protocol",
        "ip": "Internet Protocol", "http": "HyperText Transfer Protocol",
        "dns": "Domain Name System", "nat": "Network Address Translation",
        "mac": "Media Access Control", "arp": "Address Resolution Protocol",
    },
    "CNS": {
        "rsa": "RSA Public Key Cryptography", "aes": "Advanced Encryption Standard",
        "des": "Data Encryption Standard", "sha": "Secure Hash Algorithm",
        "pki": "Public Key Infrastructure", "ca": "Certificate Authority",
        "vpn": "Virtual Private Network",
    },
    "DBMS": {
        "er": "Entity Relationship", "1nf": "First Normal Form",
        "2nf": "Second Normal Form", "3nf": "Third Normal Form",
        "bcnf": "Boyce-Codd Normal Form", "sql": "Structured Query Language",
        "dml": "Data Manipulation Language", "ddl": "Data Definition Language",
        "acid": "Atomicity Consistency Isolation Durability",
    },
    "OS": {
        "fcfs": "First Come First Served scheduling", "sjf": "Shortest Job First",
        "rr": "Round Robin scheduling", "tlb": "Translation Lookaside Buffer",
        "mmu": "Memory Management Unit", "pcb": "Process Control Block",
        "ipc": "Inter-Process Communication", "vm": "Virtual Memory",
    },
    "DAA Notes": {
        "da": "Design and Analysis of Algorithms", "daa": "Design and Analysis of Algorithms",
        "dp": "Dynamic Programming", "bfs": "Breadth First Search",
        "dfs": "Depth First Search", "mst": "Minimum Spanning Tree",
        "tsp": "Travelling Salesman Problem",
    },
    "Software Engineering": {
        "sdlc": "Software Development Life Cycle", "srs": "Software Requirements Specification",
        "uml": "Unified Modelling Language", "dfd": "Data Flow Diagram",
        "er": "Entity Relationship Diagram",
    },
    "DPPM": {
        "dppm": "Data Preparation and Pattern Mining",
        "eda": "Exploratory Data Analysis",
        "pca": "Principal Component Analysis",
    }
}

SUBJECT_NAME_ABBREV = {
    "cn": "Computer Networks",
    "cns": "Cryptography and Network Security",
    "cd": "Compiler Design",
    "ai": "Artificial Intelligence",
    "daa": "Design and Analysis of Algorithms",
    "dbms": "Database Management Systems",
    "os": "Operating Systems",
    "ml": "Machine Learning",
    "nnd": "Neural Networks and Deep Learning",
    "nndl": "Neural Networks and Deep Learning",
    "dl": "Deep Learning",
    "nlp": "Natural Language Processing",
    "flat": "Formal Languages and Automata Theory",
    "devops": "DevOps",
    "se": "Software Engineering",
    "stm": "Software Testing Methodologies",
    "dppm": "Data Preparation and Pattern Mining",
    "acs": "Advanced Communication Systems",
    "coa": "Computer Organization and Architecture",
    "sna": "Social Network Analysis",
    "befa": "Business Economics and Financial Analysis",
    "msf": "Management Science and Finance",
    "tqm": "Total Quality Management",
    "poe": "Principles of Economics",
}


def is_short_query(query: str) -> bool:
    """Returns True if the query is a single abbreviation or very short query (<= 3 words)."""
    words=query.strip().split()
    return len(words)<=3 or (len(words)<=5 and all(len(w)<=5 for w in words))


def _session_path(session_id: str,sessions_dir: str="./chat_history_sessions") -> str:
    os.makedirs(sessions_dir,exist_ok=True)
    return os.path.join(sessions_dir,f"{session_id}.json")

def get_all_sessions(sessions_dir: str="./chat_history_sessions") -> List[Dict[str,Any]]:
    """Retrieve all saved chat sessions sorted by timestamp."""
    os.makedirs(sessions_dir,exist_ok=True)
    sessions=[]
    for fname in os.listdir(sessions_dir):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(sessions_dir,fname),"r",encoding="utf-8") as f:
                    sessions.append(json.load(f))
            except Exception:
                pass
    sessions.sort(key=lambda x: x.get("timestamp",0),reverse=True)
    return sessions

def load_session(session_id: str,sessions_dir: str ="./chat_history_sessions") -> Optional[Dict[str, Any]]:
    """Load a specific chat session by session_id."""
    p=_session_path(session_id,sessions_dir)
    if os.path.exists(p):
        try:
            with open(p,"r",encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_session(session_id: str,title: str,messages: List[Dict[str,Any]],subject: str="All Subjects",sessions_dir: str="./chat_history_sessions"):
    """Save or update a chat session to JSON file."""
    data={
        "session_id":session_id,
        "title":title,
        "subject":subject,
        "messages":messages,
        "timestamp":time.time()
    }
    p=_session_path(session_id,sessions_dir)
    try:
        with open(p,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
    except Exception as e:
        print(f"Error saving session {session_id}: {e}")

def delete_session(session_id: str,sessions_dir: str="./chat_history_sessions"):
    """Delete a session file by ID."""
    p=_session_path(session_id,sessions_dir)
    if os.path.exists(p):
        try:
            os.remove(p)
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")