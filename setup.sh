#!/bin/bash
# ============================================================
#  Setup Script (Mac / Linux)
#  Python 3.13 compatible
#  Run: bash setup.sh
# ============================================================

echo ""
echo "===================================================="
echo "  Setup  (Python 3.13)"
echo "===================================================="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found."
    echo "Install it from https://python.org"
    exit 1
fi

PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
echo "Done."
echo ""

# Activate
source venv/bin/activate

# Upgrade pip (important for Python 3.13 wheel resolution)
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install deps
echo "Installing dependencies (2-5 minutes)..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Some packages failed."
    echo "Try manually:"
    echo "  pip install 'nibabel>=5.3.0' 'numpy>=2.0.0' scipy 'scikit-image>=0.26.0' scikit-learn trimesh matplotlib tqdm"
    exit 1
fi

echo ""
echo "===================================================="
echo "  Setup COMPLETE!"
echo "===================================================="
echo ""
echo "Next steps:"
echo "  1. Put ACDC data in: data/training/patient001/"
echo "  2. Run: source venv/bin/activate"
echo "  3. Run: python run_all.py"
echo ""
