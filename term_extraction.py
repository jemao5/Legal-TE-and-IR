"""
Term extraction module using The_Termolator.

This module provides functions to extract domain-specific terms from patent abstracts
using The_Termolator system. It implements per-patent term extraction with a sampled
background corpus for efficient processing.
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
import pickle
import random
from typing import Dict, List, Tuple, Optional
import os


def prepare_termolator_input(
    patent_id: str,
    abstract: str,
    background_patents: List[Tuple[str, str]],
    temp_dir: Path,
) -> Tuple[Path, Path, Path]:
    """
    Prepare input files for The_Termolator.
    
    Creates temporary text files for foreground (single patent) and background
    (sampled patents), and generates file lists required by The_Termolator.
    
    Args:
        patent_id: ID of the patent to extract terms from
        abstract: Abstract text of the patent
        background_patents: List of (patent_id, abstract) tuples for background corpus
        temp_dir: Temporary directory for storing files
        
    Returns:
        Tuple of (foreground_list_path, background_list_path, output_base_path)
    """
    # Create subdirectories
    foreground_dir = temp_dir / "foreground"
    background_dir = temp_dir / "background"
    foreground_dir.mkdir(exist_ok=True)
    background_dir.mkdir(exist_ok=True)
    
    # Write foreground file (single patent)
    foreground_file = foreground_dir / f"{patent_id}.txt"
    foreground_file.write_text(abstract, encoding='utf-8')
    
    # Write background files
    background_files = []
    for bg_id, bg_abstract in background_patents:
        bg_file = background_dir / f"{bg_id}.txt"
        bg_file.write_text(bg_abstract, encoding='utf-8')
        background_files.append(bg_file)
    
    # Create file lists (one file per line, absolute paths)
    foreground_list = temp_dir / "foreground.list"
    with open(foreground_list, 'w', encoding='utf-8') as f:
        f.write(str(foreground_file.absolute()) + '\n')
    
    background_list = temp_dir / "background.list"
    with open(background_list, 'w', encoding='utf-8') as f:
        for bg_file in background_files:
            f.write(str(bg_file.absolute()) + '\n')
    
    # Output base name (without extension)
    output_base = temp_dir / f"{patent_id}_terms"
    
    return foreground_list, background_list, output_base


def run_termolator(
    foreground_list: Path,
    background_list: Path,
    extension: str,
    output_name: Path,
    termolator_dir: Path,
    process_background: bool = True,
    use_web_score: bool = False,
    max_terms: int = 30000,
    top_n: int = 5000,
    additional_topics: str = "False",
    skip_foreground_preprocessing: bool = False,
    general_filename: str = "False",
    background_cache: Optional[Path] = None,
    min_probability: str = "-.2",
) -> Path:
    """
    Run The_Termolator shell script to extract terms.
    
    Args:
        foreground_list: Path to file listing foreground documents
        background_list: Path to file listing background documents
        extension: File extension (.txt, .XML, etc.)
        output_name: Base name for output files (without extension)
        termolator_dir: Path to The_Termolator directory
        process_background: If True, process background files; if False, load from cache
        use_web_score: If True, use web-based scoring (slower but more accurate)
        max_terms: Maximum number of terms to consider
        top_n: Number of top terms to keep in final output
        additional_topics: Additional topic areas (e.g., "legal+finance") or "False"
        skip_foreground_preprocessing: If True, skip foreground preprocessing
        general_filename: General filename for shared resources or "False"
        background_cache: Path to background cache file (.pkl) or None
        min_probability: Minimum probability threshold ("-.2" for patents, "-1" for normal, or "False")
        
    Returns:
        Path to the output .out_term_list file
        
    Raises:
        subprocess.CalledProcessError: If The_Termolator execution fails
        FileNotFoundError: If output file is not created
    """
    termolator_script = termolator_dir / "run_termolator.sh"
    
    if not termolator_script.exists():
        raise FileNotFoundError(f"The_Termolator script not found at {termolator_script}")
    
    # Prepare arguments
    if background_cache:
        background_cache = Path(background_cache).resolve()
        background_cache.parent.mkdir(parents=True, exist_ok=True)
        background_cache_str = str(background_cache)
    else:
        background_cache_str = "False"
    
    # Use bash to run the shell script
    args = [
        "bash",
        str(termolator_script.absolute()),
        str(foreground_list.absolute()),
        str(background_list.absolute()),
        extension,
        str(output_name.absolute()),
        "True" if process_background else "False",
        "True" if use_web_score else "False",
        str(max_terms),
        str(top_n),
        str(termolator_dir.absolute()),
        additional_topics,
        "True" if skip_foreground_preprocessing else "False",
        general_filename,
        background_cache_str,
        min_probability,
    ]
    
    # Run The_Termolator
    print(f"Running The_Termolator for {output_name.name}...")
    
    # Set up environment to use venv Python
    import os
    env = os.environ.copy()
    venv_python = Path(__file__).parent / ".venv" / "bin" / "python3"
    
    if venv_python.exists():
        # Set PYTHON environment variable that scripts can use
        env["PYTHON"] = str(venv_python.absolute())
        
        # Also prepend venv/bin to PATH as backup
        venv_bin = venv_python.parent
        current_path = env.get("PATH", "")
        env["PATH"] = str(venv_bin.absolute()) + os.pathsep + current_path
        
        print(f"Using venv Python from: {venv_python}")
        print(f"Set PYTHON={env['PYTHON']}")
    else:
        print("Warning: .venv/bin/python3 not found, using system Python")
    
    try:
        result = subprocess.run(
            args,
            cwd=termolator_dir,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        print(f"The_Termolator completed for {output_name.name}")
        
        # Print output for debugging
        if result.stdout:
            stdout_lines = result.stdout.strip().split('\n')
            # Look for key messages
            key_lines = [l for l in stdout_lines if 'all_terms' in l or 'distributional' in l or 'scored_output' in l or 'out_term_list' in l]
            if key_lines:
                print("Key messages from The_Termolator:")
                for line in key_lines:
                    print(f"  {line}")
            elif len(stdout_lines) > 3:
                print("Last 3 lines of stdout:")
                for line in stdout_lines[-3:]:
                    print(f"  {line}")
        
        if result.stderr:
            stderr_lines = result.stderr.strip().split('\n')
            error_lines = [l for l in stderr_lines if 'Error' in l or 'Traceback' in l or 'FileNotFound' in l or 'Exception' in l]
            if error_lines:
                print("Errors in stderr:")
                for line in error_lines[-15:]:
                    print(f"  {line}")
                    
    except subprocess.CalledProcessError as e:
        print(f"Error running The_Termolator: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    
    # Check for output file
    output_file = Path(f"{output_name}.out_term_list")
    if not output_file.exists():
        # Check if .all_terms or .scored_output exists (pipeline may have stopped early)
        all_terms_file = Path(f"{output_name}.all_terms")
        scored_output_file = Path(f"{output_name}.scored_output")
        
        if all_terms_file.exists():
            print(f"Warning: .out_term_list not found, but .all_terms exists ({all_terms_file.stat().st_size} bytes)")
            print(f"  The_Termolator may have stopped at distributional component stage")
        elif scored_output_file.exists():
            print(f"Warning: .out_term_list not found, but .scored_output exists ({scored_output_file.stat().st_size} bytes)")
            print(f"  The_Termolator may have stopped at filtering stage")
        else:
            # Check for any output files
            output_dir = output_name.parent
            output_prefix = output_name.name
            related_files = list(output_dir.glob(f"{output_prefix}*"))
            if related_files:
                print(f"Found related files: {[f.name for f in related_files[:5]]}")
        
        raise FileNotFoundError(
            f"Expected output file not found: {output_file}. "
            f"The_Termolator may have failed. Check logs above."
        )
    
    return output_file


def parse_termolator_output(out_term_list_path: Path) -> List[str]:
    """
    Parse The_Termolator output file to extract terms.
    
    The .out_term_list file format is:
    - Each line contains a term lemma followed by variants, separated by tabs
    - Example: "glucocorticoid receptor\tglucocorticoid receptors\tgr"
    
    Args:
        out_term_list_path: Path to the .out_term_list file
        
    Returns:
        List of unique terms (lemmas and variants combined)
    """
    terms = []
    
    if not out_term_list_path.exists():
        print(f"Warning: Output file not found: {out_term_list_path}")
        return terms
    
    with open(out_term_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by tabs to get lemma and variants
            term_parts = line.split('\t')
            # Add all variants (lemma is first, variants follow)
            for term in term_parts:
                term = term.strip()
                if term and term not in terms:
                    terms.append(term)
    
    return terms


def extract_terms_for_patent(
    patent_id: str,
    abstract: str,
    background_patents: List[Tuple[str, str]],
    termolator_dir: Path,
    temp_base_dir: Optional[Path] = None,
    process_background: bool = True,
    background_cache: Optional[Path] = None,
    keep_temp_files: bool = False,
    **termolator_kwargs,
) -> List[str]:
    """
    Extract terms for a single patent using The_Termolator.
    
    Args:
        patent_id: ID of the patent
        abstract: Abstract text of the patent
        background_patents: List of (patent_id, abstract) tuples for background
        termolator_dir: Path to The_Termolator directory
        temp_base_dir: Base directory for temporary files (default: system temp)
        process_background: If True, process background; if False, use cache
        background_cache: Path to background cache file (for reuse)
        keep_temp_files: If True, keep temporary files for debugging
        **termolator_kwargs: Additional arguments for run_termolator()
        
    Returns:
        List of extracted terms
    """
    # Create temporary directory
    if temp_base_dir:
        temp_dir = temp_base_dir / f"termolator_{patent_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"termolator_{patent_id}_"))
    
    try:
        # Prepare input files
        foreground_list, background_list, output_base = prepare_termolator_input(
            patent_id, abstract, background_patents, temp_dir
        )
        
        # Set default termolator arguments
        default_kwargs = {
            'extension': '.txt',
            'use_web_score': False,
            'max_terms': 30000,
            'top_n': 5000,
            'additional_topics': 'False',
            'skip_foreground_preprocessing': False,
            'general_filename': 'False',
            'min_probability': '-.2',  # Patent-specific setting
        }
        default_kwargs.update(termolator_kwargs)
        
        # Run The_Termolator
        output_file = run_termolator(
            foreground_list,
            background_list,
            output_name=output_base,
            termolator_dir=termolator_dir,
            process_background=process_background,
            background_cache=background_cache,
            **default_kwargs,
        )
        
        # Parse output
        terms = parse_termolator_output(output_file)
        
        return terms
    
    finally:
        # Cleanup temporary files
        if not keep_temp_files and temp_dir.exists():
            shutil.rmtree(temp_dir)


def sample_background_corpus(
    all_patents: Dict[str, str],
    exclude_patent_id: Optional[str] = None,
    sample_size: int = 1000,
    random_seed: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Sample a background corpus from all patents.
    
    This function creates a FIXED background corpus that will be reused
    across all patent extractions. Using the same random_seed ensures
    the same corpus is generated every time, providing consistency.
    
    Args:
        all_patents: Dictionary of {patent_id: abstract}
        exclude_patent_id: Patent ID to exclude from background (usually the foreground patent)
        sample_size: Number of patents to sample (default: 3000, larger is better)
        random_seed: Random seed for reproducibility (default: 42, ensures fixed corpus)
        
    Returns:
        List of (patent_id, abstract) tuples for background corpus
    """
    # Filter out the excluded patent if specified
    available_patents = {
        pid: abstract
        for pid, abstract in all_patents.items()
        if pid != exclude_patent_id
    }
    
    if len(available_patents) == 0:
        raise ValueError("No patents available for background corpus")
    
    # Sample
    if random_seed is not None:
        random.seed(random_seed)
    
    # If we have fewer patents than sample_size, use all available
    sample_size = min(sample_size, len(available_patents))
    
    sampled_ids = random.sample(list(available_patents.keys()), sample_size)
    return [(pid, available_patents[pid]) for pid in sampled_ids]


def extract_terms_for_patents(
    patents: Dict[str, str],
    termolator_dir: Path,
    background_sample_size: int = 1000,
    background_cache_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    random_seed: int = 42,
    process_background_once: bool = True,
    **termolator_kwargs,
) -> Dict[str, List[str]]:
    """
    Extract terms for multiple patents using a shared background corpus.
    
    This function implements the recommended strategy: per-patent extraction
    with a fixed sampled background corpus for efficiency and consistency.
    
    The background corpus is:
    - Large (default 1000 patents) for better statistical comparison
    - Fixed (same random seed ensures same corpus across runs)
    - Reused (processed once, cached for all subsequent patents)
    
    Args:
        patents: Dictionary of {patent_id: abstract}
        termolator_dir: Path to The_Termolator directory
        background_sample_size: Number of patents to sample for background (default: 1000)
        background_cache_path: Path to save/load background cache
        output_path: Path to save extracted terms (pickle file)
        random_seed: Random seed for background sampling (ensures same corpus across runs)
        process_background_once: If True, process background once and reuse cache
        **termolator_kwargs: Additional arguments for run_termolator()
        
    Returns:
        Dictionary of {patent_id: [list of terms]}
    """
    print(f"Extracting terms for {len(patents)} patents...")
    
    # Sample background corpus (FIXED - same for all patents and all runs)
    # Using fixed random_seed ensures the same background corpus is used
    # across different runs, providing consistency in term extraction
    print(f"Sampling FIXED background corpus ({background_sample_size} patents, seed={random_seed})...")
    background_corpus = sample_background_corpus(
        patents,
        sample_size=background_sample_size,
        random_seed=random_seed,
    )
    print(f"Background corpus: {len(background_corpus)} patents (will be reused for all patents)")
    
    # Process background once if requested
    background_cache = None
    first_patent_terms = None
    if process_background_once and background_cache_path:
        # First run: process background and create cache
        # We'll use the first patent as a dummy foreground to process background
        first_patent_id = list(patents.keys())[0]
        first_abstract = patents[first_patent_id]
        
        print("Processing background corpus (one-time setup)...")
        # This will create the background cache and also extract terms for the first patent
        first_patent_terms = extract_terms_for_patent(
            first_patent_id,
            first_abstract,
            background_corpus,
            termolator_dir,
            process_background=True,
            background_cache=background_cache_path,
            **termolator_kwargs,
        )
        background_cache = background_cache_path
        print(f"Background cache saved to {background_cache_path}")
        print(f"  Also extracted {len(first_patent_terms)} terms for {first_patent_id}")
        
        # Skip first patent in loop since we already processed it
        patents_to_process = {k: v for k, v in patents.items() if k != first_patent_id}
    else:
        patents_to_process = patents
    
    # Extract terms for each patent
    patent_terms = {}
    temp_base_dir = Path("temp_termolator")  # Use a common temp directory
    
    # If we processed background with first patent, add its terms to results
    if first_patent_terms is not None:
        first_patent_id = list(patents.keys())[0]
        patent_terms[first_patent_id] = first_patent_terms
        start_idx = 2
    else:
        start_idx = 1
    
    for i, (patent_id, abstract) in enumerate(patents_to_process.items(), start_idx):
        print(f"\n[{i}/{len(patents)}] Extracting terms for patent {patent_id}...")
        
        try:
            terms = extract_terms_for_patent(
                patent_id,
                abstract,
                background_corpus,
                termolator_dir,
                temp_base_dir=temp_base_dir,
                process_background=False if background_cache else True,
                background_cache=background_cache,
                **termolator_kwargs,
            )
            
            patent_terms[patent_id] = terms
            print(f"  Extracted {len(terms)} terms")
            
        except Exception as e:
            print(f"  Error extracting terms for {patent_id}: {e}")
            patent_terms[patent_id] = []  # Empty list on error
    
    # Save results
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(patent_terms, f)
        print(f"\nSaved extracted terms to {output_path}")
    
    # Cleanup temp directory
    if temp_base_dir.exists():
        shutil.rmtree(temp_base_dir)
    
    return patent_terms


def load_patent_terms(terms_path: Path) -> Dict[str, List[str]]:
    """
    Load extracted terms from a pickle file.
    
    Args:
        terms_path: Path to the pickle file containing patent terms
        
    Returns:
        Dictionary of {patent_id: [list of terms]}
    """
    with open(terms_path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="Extract terms from patent abstracts using The_Termolator"
    )
    parser.add_argument(
        "--abstracts",
        type=Path,
        default=Path("data/filtered_abstracts.tsv"),
        help="Path to TSV file with patent abstracts",
    )
    parser.add_argument(
        "--termolator-dir",
        type=Path,
        default=Path("The_Termolator"),
        help="Path to The_Termolator directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/patent_terms.pickle"),
        help="Output path for extracted terms (pickle file)",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=1000,
        help="Number of patents to sample for background corpus (default: 1000)",
    )
    parser.add_argument(
        "--background-cache",
        type=Path,
        default=Path("data/termolator_background_cache.pkl"),
        help="Path to background cache file",
    )
    parser.add_argument(
        "--max-patents",
        type=int,
        default=None,
        help="Maximum number of patents to process (for testing)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for background sampling (ensures same background corpus across runs)",
    )
    
    args = parser.parse_args()
    
    # Load abstracts
    print(f"Loading abstracts from {args.abstracts}...")
    patents = {}
    with open(args.abstracts, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip().split('\t')
            line = [elem.strip('"') for elem in line]
            if len(line) > 1:
                patents[line[0]] = line[1]
    
    if args.max_patents:
        patents = dict(list(patents.items())[:args.max_patents])
        print(f"Limited to {args.max_patents} patents for testing")
    
    print(f"Loaded {len(patents)} patents")
    
    # Extract terms
    patent_terms = extract_terms_for_patents(
        patents,
        args.termolator_dir,
        background_sample_size=args.background_size,
        background_cache_path=args.background_cache,
        output_path=args.output,
        random_seed=args.random_seed,
    )
    
    print(f"\nTerm extraction complete!")
    print(f"Total patents processed: {len(patent_terms)}")
    print(f"Average terms per patent: {sum(len(terms) for terms in patent_terms.values()) / len(patent_terms):.1f}")

