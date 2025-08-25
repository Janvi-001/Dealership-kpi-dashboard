import os
import re
import subprocess

# Simple regex to capture "import xxx" and "from xxx import"
IMPORT_RE = re.compile(r'^(?:from|import)\s+([a-zA-Z0-9_]+)')

# Mapping: common import names → pip package names
PACKAGE_MAP = {
    "sklearn": "scikit-learn",
    "statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "numpy": "numpy",
    "streamlit": "streamlit",
    # add more mappings if needed
}

def find_imports(py_file):
    imports = set()
    with open(py_file, "r", encoding="utf-8") as f:
        for line in f:
            match = IMPORT_RE.match(line.strip())
            if match:
                mod = match.group(1).split(".")[0]
                imports.add(mod)
    return imports

def generate_requirements(root_dir="."):
    all_imports = set()
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                all_imports |= find_imports(os.path.join(root, file))

    # Map module names to pip packages
    packages = {PACKAGE_MAP.get(mod, mod) for mod in all_imports}

    # Try to pin versions (optional)
    pinned = []
    for pkg in packages:
        try:
            version = subprocess.check_output(
                ["pip", "show", pkg], text=True
            )
            for line in version.splitlines():
                if line.startswith("Version:"):
                    v = line.split(":")[1].strip()
                    pinned.append(f"{pkg}=={v}")
                    break
        except Exception:
            pinned.append(pkg)

    with open("requirements.txt", "w") as f:
        f.write("\n".join(sorted(pinned)))

    print("✅ requirements.txt generated with packages:")
    print("\n".join(sorted(pinned)))

if __name__ == "__main__":
    generate_requirements(".")
