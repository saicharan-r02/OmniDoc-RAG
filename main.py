import sys
import os
import argparse

def run_streamlit():
    """Launch the Streamlit web interface."""
    import subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    print("🚀 Starting OmniDoc AI Streamlit Application...")
    subprocess.run(cmd)

def run_ingestion(data_dir: str = "./PDF_Data", subject: str = None, rebuild: bool = False):
    """Run incremental or explicit safe rebuild ingestion into ChromaDB."""
    from src.ingestion.loader import process_directory_incrementally
    mode = "rebuild" if rebuild else "incremental"
    print(f"Starting {mode} ingestion from '{data_dir}'...")
    process_directory_incrementally(root_path=data_dir, subject_filter=subject, rebuild=rebuild)

def main():
    parser = argparse.ArgumentParser(description="OmniDoc-RAG: Engineering Academic Assistant")
    parser.add_argument(
        "--mode",
        choices=["app", "ingest", "test"],
        default="app",
        help="Execution mode: 'app' (default: Streamlit UI), 'ingest' (process PDFs), 'test' (run test suite)"
    )
    parser.add_argument("--data-dir", default="./PDF_Data", help="Path to PDF directory for ingestion")
    parser.add_argument("--subject", help="Rebuild only this PDF_Data subject folder")
    parser.add_argument("--rebuild", action="store_true", help="Clear matching vector records before ingestion; never deletes source files")

    args = parser.parse_args()

    if args.mode=="app":
        run_streamlit()
    elif args.mode=="ingest":
        run_ingestion(args.data_dir, subject=args.subject, rebuild=args.rebuild)
    elif args.mode=="test":
        import unittest
        loader=unittest.TestLoader()
        suite=loader.discover("tests")
        runner=unittest.TextTestRunner(verbosity=2)
        runner.run(suite)

if __name__=="__main__":
    main()