"""
Test script to run all 4 retrieval methods on a small subset.

Methods tested:
1. TF-IDF
2. BM25
3. Embedding-based (full abstracts)
4. Term extraction + Embedding-based

Note: This script assumes dependencies are installed. If you see import errors,
install dependencies using:
  pip install -r requirements.txt
or
  uv sync
"""

import subprocess
import sys
from pathlib import Path
import time
import os


def run_command(cmd, description, use_shell=False):
    """Run a command and print timing information."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd) if not use_shell else cmd}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=3600,  # 1 hour timeout
            shell=use_shell
        )
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.2f} seconds")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed after {elapsed:.2f} seconds")
        print(f"Error: {e}")
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        return False, elapsed


def check_data_files():
    """Check if required data files exist."""
    required_files = [
        "data/filtered_abstracts.tsv",
        "data/filtered_citations.tsv",
        "data/labelled_ids.pickle",
        "data/filing_dates.pickle",
    ]
    
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print("⚠ Missing required data files:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease run data filtering first:")
        print("  python data_filtering.py --max-patents 50")
        return False
    return True


def check_python_environment():
    """Check if required Python packages are available."""
    try:
        import numpy
        import nltk
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError as e:
        print(f"⚠ Missing Python package: {e}")
        print("\nPlease install dependencies:")
        print("  pip install numpy nltk sentence-transformers scikit-learn torch")
        print("or")
        print("  uv sync")
        return False


def find_python_executable():
    """Find the appropriate Python executable (prefer venv if available)."""
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        return str(venv_python.absolute())
    
    # Try uv run
    if Path("/home/vikram/.local/bin/uv").exists():
        return "uv run python"
    
    # Fall back to system python3
    return sys.executable


def main():
    """Run all 4 methods on a small subset."""
    print("="*70)
    print("Testing All 4 Retrieval Methods on Small Subset")
    print("="*70)
    
    # Find Python executable
    python_exe = find_python_executable()
    print(f"Using Python: {python_exe}\n")
    
    # Check if data files exist
    if not check_data_files():
        return
    
    # Configuration
    max_patents = 50  # Small subset for testing
    num_queries = 10  # Small number of queries
    recall_k = 100
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    results = {}
    
    # Method 1: TF-IDF
    print("\n" + "="*70)
    print("METHOD 1: TF-IDF")
    print("="*70)
    success, elapsed = run_command(
        [
            python_exe, "tfidf.py",
            "--num-queries", str(num_queries),
            "--recall-k", str(recall_k),
            "--force-index",
        ],
        "TF-IDF Retrieval"
    )
    results["TF-IDF"] = {"success": success, "time": elapsed}
    
    # Method 2: BM25
    print("\n" + "="*70)
    print("METHOD 2: BM25")
    print("="*70)
    success, elapsed = run_command(
        [
            python_exe, "bm25_system.py",
            "--num-queries", str(num_queries),
            "--recall-k", str(recall_k),
            "--force-index",
        ],
        "BM25 Retrieval"
    )
    results["BM25"] = {"success": success, "time": elapsed}
    
    # Method 3: Embedding-based (full abstracts)
    print("\n" + "="*70)
    print("METHOD 3: Embedding-based (Full Abstracts)")
    print("="*70)
    success, elapsed = run_command(
        [
            python_exe, "embedding_rankings.py",
            "--num-queries", str(num_queries),
            "--recall-k", str(recall_k),
            "--force-index",
        ],
        "Full Abstract Embedding Retrieval"
    )
    results["Embedding (Full)"] = {"success": success, "time": elapsed}
    
    # Method 4: Term extraction + Embedding
    print("\n" + "="*70)
    print("METHOD 4: Term Extraction + Embedding")
    print("="*70)
    
    # Check if The_Termolator exists
    termolator_dir = Path("The_Termolator")
    if not (termolator_dir / "run_termolator.sh").exists():
        print("⚠ The_Termolator not found. Skipping term extraction method.")
        print(f"   Expected: {termolator_dir / 'run_termolator.sh'}")
        results["Term + Embedding"] = {"success": False, "time": 0}
    else:
        # Step 4a: Term extraction (if not already done)
        patent_terms_path = Path("data/patent_terms.pickle")
        if not patent_terms_path.exists():
            print("\nStep 4a: Running term extraction...")
            print("⚠ WARNING: Term extraction can take a very long time!")
            print("   Consider running it separately first:")
            print(f"   python3 term_extraction.py --max-patents {max_patents} --background-size 1000")
            
            # Ask user if they want to continue (for now, we'll skip)
            print("\n   Skipping term extraction in automated test.")
            print("   Please run term extraction manually first, then re-run this test.")
            results["Term Extraction"] = {"success": False, "time": 0}
            results["Term + Embedding"] = {"success": False, "time": 0}
        else:
            print("Term extraction already completed. Using cached results.")
            results["Term Extraction"] = {"success": True, "time": 0}
            
            # Step 4b: Term-based embedding retrieval
            print("\nStep 4b: Running term-based embedding retrieval...")
            success, elapsed = run_command(
                [
                    python_exe, "term_embedding_rankings.py",
                    "--num-queries", str(num_queries),
                    "--recall-k", str(recall_k),
                    "--force-index",
                ],
                "Term-based Embedding Retrieval"
            )
            results["Term + Embedding"] = {"success": success, "time": elapsed}
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"{'Method':<30} {'Status':<15} {'Time (s)':<15}")
    print("-" * 70)
    
    for method, result in results.items():
        status = "✓ Success" if result["success"] else "✗ Failed"
        time_str = f"{result['time']:.2f}" if result["time"] > 0 else "N/A"
        print(f"{method:<30} {status:<15} {time_str:<15}")
    
    total_time = sum(r["time"] for r in results.values() if r["time"] > 0)
    print(f"\nTotal time: {total_time:.2f} seconds")
    
    # Check output files
    print("\n" + "="*70)
    print("OUTPUT FILES CHECK")
    print("="*70)
    output_files = {
        "TF-IDF": "data/tfidf_rankings.tsv",
        "BM25": "data/bm25_rankings.tsv",
        "Embedding (Full)": "data/embedding_rankings.tsv",
        "Term + Embedding": "data/term_embedding_rankings.tsv",
    }
    
    for method, filepath in output_files.items():
        path = Path(filepath)
        if path.exists():
            size = path.stat().st_size
            print(f"✓ {method}: {filepath} ({size:,} bytes)")
        else:
            print(f"✗ {method}: {filepath} (not found)")


if __name__ == "__main__":
    main()

