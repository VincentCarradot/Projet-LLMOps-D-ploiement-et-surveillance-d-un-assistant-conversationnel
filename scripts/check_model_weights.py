"""Hook pre-commit: bloque les poids de modele dans la zone de staging."""

from __future__ import annotations

import re
import subprocess
import sys

MOTIF_POIDS = re.compile(r"\.(pt|bin|safetensors|onnx)$", re.IGNORECASE)


def main() -> int:
    resultat = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        check=False,
        capture_output=True,
        text=True,
    )
    if resultat.returncode != 0:
        print(resultat.stderr, file=sys.stderr)
        return resultat.returncode

    fichiers_interdits = [
        ligne.strip()
        for ligne in resultat.stdout.splitlines()
        if MOTIF_POIDS.search(ligne.strip())
    ]
    if fichiers_interdits:
        print("Poids de modele interdits dans Git:", file=sys.stderr)
        for chemin in fichiers_interdits:
            print(f"  - {chemin}", file=sys.stderr)
        print("Utilisez un cache HuggingFace, un volume Docker ou Git LFS.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

