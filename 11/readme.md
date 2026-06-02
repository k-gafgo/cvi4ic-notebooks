# Clone Repository
`git clone --recursive https://github.com/schedldave/cvi4ic-notebooks.git`

# Install UV if necessary (optional)
`curl -LsSf https://astral.sh/uv/install.sh | sh`

# Activate workspace and register jupyter kernel 
```
cd 11
uv sync
source .venv/bin/activate
python -m ipykernel install --user --name CV1112 --display-name "Python (CV1112)"
``` 

# Select kernel in notebook
```
Kernel
Select Another Kernel
Azure ML compute Instance
Python (CV1112)
```