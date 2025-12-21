#!/bin/bash
echo "Initializing Git Repository..."
git init

echo "Adding files (respecting .gitignore)..."
git add .

echo "Committing..."
git commit -m "Initial commit: Safe push with secrets excluded"

echo "Renaming branch to main..."
git branch -M main

echo "Adding remote..."
git remote add origin https://github.com/canukguy1974/avatar-pipeline.git

echo "Pushing to GitHub..."
git push -u origin main
