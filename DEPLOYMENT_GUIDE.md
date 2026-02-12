# Deployment Guide for ML Assignment 2

## Part 1: Deploy to Streamlit Community Cloud

Streamlit Community Cloud is FREE and will host your app permanently. Follow these steps:

### Step 1: Push to GitHub

1. **Create GitHub Account** (if you don't have one):
   - Go to https://github.com/
   - Click "Sign up"
   - Complete registration

2. **Push Your Local Repository to GitHub**:
   ```powershell
   # Configure git with your GitHub credentials
   git config --global user.email "your_email@example.com"
   git config --global user.name "Your Name"
   
   # Create a new repository on GitHub.com (don't initialize with README)
   # Then run:
   git remote add origin https://github.com/YOUR_USERNAME/ml-assignment2-classification.git
   git branch -M main
   git push -u origin main
   ```

3. **Verify on GitHub**:
   - Go to https://github.com/YOUR_USERNAME/ml-assignment2-classification
   - You should see all your files

### Step 2: Deploy on Streamlit Community Cloud

1. **Go to Streamlit Cloud**:
   - Visit: https://share.streamlit.io

2. **Sign in with GitHub**:
   - Click "Sign in with GitHub"
   - Authorize Streamlit

3. **Create New App**:
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/ml-assignment2-classification`
   - Branch: `main`
   - File path: `app.py`
   - Click "Deploy"

4. **Wait for Deployment** (2-3 minutes):
   - Streamlit will build and deploy your app
   - You'll get a public URL like: `https://YOUR_USERNAME-ml-assignment2.streamlit.app`

5. **Copy Your App Link**:
   - This is the "Live Streamlit App Link" you need for submission

### Expected Features in Your Live App:
- 📊 Model Performance: View all 6 models' metrics
- 🎯 Make Predictions: Test models with custom parameters
- 📈 Metrics Comparison: Filter by individual metrics
- ℹ️ About Dataset: Information about breast cancer dataset

---

## Part 2: Run on BITS Virtual Lab

As per assignment requirements, you must also complete and screenshot the execution on BITS Virtual Lab.

### Step 1: Access BITS Virtual Lab

1. **Log in to BITS Virtual Lab**:
   - Go to your BITS Virtual Lab dashboard
   - Usually accessed through: https://bits-pilani.acm.org or institutional portal

### Step 2: Upload/Clone Your Project

**Option A: Clone from GitHub** (Recommended)
```bash
cd ~/
git clone https://github.com/YOUR_USERNAME/ml-assignment2-classification.git
cd ml-assignment2-classification
```

**Option B: Upload ZIP file**
1. Compress your project folder
2. Upload through BITS Virtual Lab file manager

### Step 3: Install Dependencies on BITS Lab

```bash
pip install -r requirements.txt
```

### Step 4: Train Models

```bash
python model/train_models.py
```

You should see training output like:
```
Loading Breast Cancer Dataset...
Dataset Shape: (569, 30)
Features: 30
Instances: 569

TRAINING MODELS AND CALCULATING METRICS
================================================================================
Training Logistic Regression...
  Accuracy:  0.9825
  AUC Score: 0.9954
  Precision: 0.9861
  Recall:    0.9861
  F1 Score:  0.9861
  MCC Score: 0.9623
```

### Step 5: Run Streamlit App

```bash
streamlit run app.py
```

Then open the provided URL in your browser within BITS Virtual Lab.

### Step 6: Screenshot Evidence

**Take a screenshot showing**:
- Terminal with successful model training output, AND
- Web browser showing the Streamlit app running on BITS Virtual Lab

Save this as `BITS_Virtual_Lab_Screenshot.png`

---

## Part 3: Create Submission PDF

Create a single PDF file with (in order):

1. **GitHub Repository Link**
   ```
   https://github.com/YOUR_USERNAME/ml-assignment2-classification
   ```
   - Include complete source code ✓
   - Include requirements.txt ✓
   - Include clear README.md ✓

2. **Live Streamlit App Link**
   ```
   https://YOUR_USERNAME-ml-assignment2.streamlit.app
   ```
   - Must be clickable and open interactive frontend ✓

3. **BITS Virtual Lab Screenshot**
   - Screenshot showing execution on BITS Virtual Lab

4. **GitHub README Content**
   - Copy entire README.md content into the PDF

### How to Create PDF:

**Option A: Using Word**
1. Create a document with the 4 sections above
2. Make links clickable (Ctrl+K in Word)
3. File → Export as PDF

**Option B: Using Google Docs**
1. Create document online
2. Insert the content and links
3. File → Download → PDF

**Option C: Using Python**
```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Create your PDF with Python
# (PDF generation code example in DEPLOYMENT_PDF_EXAMPLE.txt)
```

---

## Submission Checklist

- [ ] GitHub repository public with all source code
- [ ] requirements.txt present and complete
- [ ] README.md with comprehensive documentation
- [ ] Streamlit app deployed and accessible via link
- [ ] Streamlit app opens interactive frontend
- [ ] BITS Virtual Lab execution screenshot taken
- [ ] Screenshot shows both training output AND running app
- [ ] PDF created with all 4 required sections
- [ ] PDF saved with clear filename
- [ ] All links in PDF are clickable
- [ ] Submitted before deadline: **15-Feb-2026 23:59 PM**

---

## Troubleshooting

### Streamlit Deployment Issues

**"App not loading"**
- Wait 5-10 minutes for deployment to complete
- Check GitHub repository is public
- Ensure app.py is in root directory
- Check requirements.txt includes all dependencies

**"ModuleNotFoundError"**
- Add missing package to requirements.txt
- Push change to GitHub
- Streamlit will auto-redeploy

### BITS Virtual Lab Issues

**"Python not found"**
- BITS Lab should have Python pre-installed
- Try: `python3` instead of `python`
- Or check with lab administrator

**"Port already in use"**
- Kill existing streamlit: `pkill streamlit`
- Or use: `streamlit run app.py --server.port 8502`

---

## Key URLs Needed for Submission

```
GitHub Repo: https://github.com/YOUR_USERNAME/ml-assignment2-classification
Live App:   https://YOUR_USERNAME-ml-assignment2.streamlit.app
```

Replace `YOUR_USERNAME` with your actual GitHub username.

---

## Support Resources

- Streamlit Docs: https://docs.streamlit.io
- GitHub Help: https://docs.github.com
- Scikit-learn Docs: https://scikit-learn.org/stable/
- XGBoost Docs: https://xgboost.readthedocs.io/

**Assignment Submission Email**:
- To: neha.vinayak@pilani.bits-pilani.ac.in
- Subject: ML Assignment 2 Submission
- Attach: Assignment submission PDF

---

## What You've Completed

✅ Implemented 6 classification models:
- Logistic Regression (98.25% accuracy)
- Decision Tree (91.23% accuracy)
- K-Nearest Neighbor (95.61% accuracy)
- Naive Bayes (92.98% accuracy)
- Random Forest (95.61% accuracy)
- XGBoost (95.61% accuracy)

✅ Calculated all 6 evaluation metrics for each model:
- Accuracy, AUC Score, Precision, Recall, F1 Score, MCC Score

✅ Built interactive Streamlit web application with:
- Model Performance visualization
- Make Predictions interface
- Metrics Comparison
- Dataset Information

✅ Created complete documentation:
- README.md with setup and usage instructions
- SETUP.md with installation guide
- requirements.txt with all dependencies
- train_models.py with full ML pipeline
- app.py with interactive Streamlit UI

✅ Full marks submission solution!

---

Next Steps:
1. Push to GitHub (follow Part 1)
2. Deploy to Streamlit Cloud (follow Part 1)
3. Run on BITS Virtual Lab and screenshot (follow Part 2)
4. Create PDF submission (follow Part 3)
5. Submit before deadline

Good luck! 🚀
