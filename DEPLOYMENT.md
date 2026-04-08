# Deployment Guide - Render

## Prerequisites
- GitHub account
- Render account (https://render.com)
- OpenAI API Key
- Tavily API Key

## Deploy to Render

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - ready for Render deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/AdaptiveRag.git
   git push -u origin main