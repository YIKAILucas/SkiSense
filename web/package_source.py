"""Bundle the complete corresponding source for this hosted adaptation."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


root = Path(__file__).resolve().parent.parent
dist = root / "dist"
dist.mkdir(exist_ok=True)
exclude = {".git", "dist", "models", "input", "output", "__pycache__", ".env"}
with ZipFile(dist / "SkiSense-source.zip", "w", ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_file() and not (exclude & set(relative.parts)) and not any(part.startswith(".venv") for part in relative.parts):
            archive.write(path, Path("SkiSense") / relative)
