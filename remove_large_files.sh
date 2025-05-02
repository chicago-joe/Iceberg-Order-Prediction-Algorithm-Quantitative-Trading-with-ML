#!/bin/bash

# Remove the large notebook files from Git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch src/neptune-optuna-hpo/refactor_optuna_data.ipynb src/neptune-optuna-hpo/optuna_plot_exporter.ipynb" \
  --prune-empty --tag-name-filter cat -- --all

# Force push to remote repository
echo "Large files have been removed from Git history."
echo "Now you can force push with: git push --force origin v2-hpo"