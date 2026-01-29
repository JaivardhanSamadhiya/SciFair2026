from pathlib import Path

ROOT = Path("SciFair2026")

DIRS = [
    "data/raw",
    "data/phages",
    "data/hosts",
    "data/features",
    "scripts",
    "models",
    "results",
    "logs"
]

def main():
    for d in DIRS:
        path = ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created {path}")

    print("\nProject structure initialized ✅")

if __name__ == "__main__":
    main()
