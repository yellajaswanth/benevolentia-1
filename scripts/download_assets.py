#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
H1_PATH = "unitree_h1"


def download_h1_assets(output_dir: Path, force: bool = False) -> None:
    assets_dir = output_dir / "assets" / "unitree_h1"
    
    if assets_dir.exists() and not force:
        print(f"Assets already exist at {assets_dir}. Use --force to overwrite.")
        return
    
    temp_dir = output_dir / ".temp_menagerie"
    
    try:
        print("Cloning MuJoCo Menagerie (sparse checkout)...")
        subprocess.run(
            ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse", MENAGERIE_REPO, str(temp_dir)],
            check=True,
            capture_output=True,
        )
        
        subprocess.run(
            ["git", "-C", str(temp_dir), "sparse-checkout", "set", H1_PATH],
            check=True,
            capture_output=True,
        )
        
        source_dir = temp_dir / H1_PATH
        if not source_dir.exists():
            raise FileNotFoundError(f"H1 assets not found in menagerie at {source_dir}")
        
        assets_dir.parent.mkdir(parents=True, exist_ok=True)
        if assets_dir.exists():
            shutil.rmtree(assets_dir)
        
        shutil.copytree(source_dir, assets_dir)
        print(f"Successfully downloaded H1 assets to {assets_dir}")
        
        print("\nH1 Model Files:")
        for f in sorted(assets_dir.iterdir()):
            print(f"  - {f.name}")
            
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(description="Download Unitree H1 assets from MuJoCo Menagerie")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Output directory for assets",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing assets",
    )
    args = parser.parse_args()
    
    try:
        download_h1_assets(args.output_dir, args.force)
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

