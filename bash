git clone https://github.com/marcdaghar/yusuf-grondona-entropy.git
cd yusuf-grondona-entropy

mkdir -p .github/workflows
mkdir -p tests

# Créer .github/workflows/test.yml
cat > .github/workflows/test.yml << 'EOF'
[COLLER LE CONTENU DU WORKFLOW YAML ICI]
EOF

# Créer tests/test_basics.py
cat > tests/test_basics.py << 'EOF'
[COLLER LE CONTENU DE test_basics.py ICI]
EOF

# Créer pyproject.toml
cat > pyproject.toml << 'EOF'
[COLLER LE CONTENU DE pyproject.toml ICI]
EOF

# Créer __init__.py vide
touch tests/__init__.py

git add .github/ tests/ pyproject.toml
git commit -m "Add CI/CD, unit tests, and pyproject.toml

- GitHub Actions workflow for automated testing (Python 3.9-3.11)
- 14 unit tests covering core models, validation, and shocks
- pyproject.toml with modern Python packaging standards"
git push origin main
