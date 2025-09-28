#!/usr/bin/env python3
"""
Upload existing ChromaDB directories to HuggingFace datasets.

Usage:
    python upload_chroma_to_hf.py
    
This script will upload all ChromaDB directories found in the congress-hf directory
to HuggingFace datasets following the hyperdemocracy naming convention.
"""

from pathlib import Path
import rich
from legisplain.chroma import upload_chroma_to_hf


def main():
    """Main function to upload all ChromaDB directories to HuggingFace datasets."""
    congress_hf_path = Path("/Users/galtay/repos/legisplain/congress-hf")
    hf_org = "hyperdemocracy"
    
    # Find all ChromaDB directories
    chroma_dirs = []
    for path in congress_hf_path.iterdir():
        if path.is_dir() and path.name.startswith("usc-chroma-"):
            chromadb_path = path / "chromadb"
            if chromadb_path.exists():
                chroma_dirs.append(chromadb_path)
    
    if not chroma_dirs:
        rich.print("[yellow]No ChromaDB directories found to upload[/yellow]")
        return
    
    rich.print(f"[blue]Found {len(chroma_dirs)} ChromaDB directories to upload:[/blue]")
    for chroma_dir in chroma_dirs:
        dataset_name = chroma_dir.parent.name
        repo_id = f"{hf_org}/{dataset_name}"
        rich.print(f"  - {dataset_name} → {repo_id}")
    
    print()
    
    # Ask for confirmation
    rich.print("[yellow]This will create public HuggingFace datasets. Continue? (y/N)[/yellow]")
    response = input().strip().lower()
    if response not in ['y', 'yes']:
        rich.print("[blue]Upload cancelled.[/blue]")
        return
    
    print()
    
    # Upload each directory
    successful_uploads = 0
    failed_uploads = 0
    
    for chroma_dir in chroma_dirs:
        dataset_name = chroma_dir.parent.name
        repo_id = f"{hf_org}/{dataset_name}"
        
        rich.print(f"[blue]Uploading: {dataset_name}[/blue]")
        rich.print(f"[blue]Local path: {chroma_dir.parent}[/blue]")
        rich.print(f"[blue]HF dataset: {repo_id}[/blue]")
        
        success = upload_chroma_to_hf(chroma_dir, hf_org)
        
        if success:
            successful_uploads += 1
            rich.print(f"[green]✓ Successfully uploaded: {dataset_name}[/green]")
            rich.print(f"[green]  View at: https://huggingface.co/datasets/{repo_id}[/green]")
        else:
            failed_uploads += 1
            rich.print(f"[red]✗ Failed to upload: {dataset_name}[/red]")
        
        print()
    
    # Summary
    rich.print("[blue]Upload Summary:[/blue]")
    rich.print(f"  Successful: {successful_uploads}")
    rich.print(f"  Failed: {failed_uploads}")
    rich.print(f"  Total: {len(chroma_dirs)}")
    
    if failed_uploads == 0:
        rich.print("[green]All ChromaDB datasets uploaded successfully! 🎉[/green]")
        rich.print("[blue]Your HF Spaces can now use these datasets with:[/blue]")
        rich.print("[blue]  from huggingface_hub import snapshot_download[/blue]")
        rich.print("[blue]  local_dir = snapshot_download(repo_id='hyperdemocracy/usc-chroma-...')[/blue]")
    else:
        rich.print(f"[yellow]{failed_uploads} uploads failed. Check the error messages above.[/yellow]")


if __name__ == "__main__":
    main()
