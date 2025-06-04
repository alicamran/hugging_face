# DLL Error Fix: Simple Solution

If you're experiencing the error: `ImportError: DLL load failed while importing lib: The specified module could not be found.`

This is likely due to issues with the Python libraries that depend on C++ DLLs on Windows. The transformers library and PyTorch can sometimes cause these problems.

## Quick Fix: Use the Simple Sentiment App

I've created a simple sentiment analysis app that doesn't depend on complex libraries like transformers or PyTorch. This should work without any DLL issues:

```bash
streamlit run simple_sentiment_app.py
```

### Features of the Simple App:
- No dependencies on transformers or PyTorch
- Works completely offline
- Uses a simple lexicon-based approach
- Shows positive and negative word counts
- Displays sentiment score and visualization

## If You Still Want to Use the Original App

To fix the DLL issues with the original Hugging Face app:

1. **Install Visual C++ Redistributable**:
   Download from: https://aka.ms/vs/16/release/vc_redist.x64.exe

2. **Create a new virtual environment**:
   ```bash
   python -m venv sentiment_env
   sentiment_env\Scripts\activate
   ```

3. **Install CPU-only PyTorch** (avoids CUDA dependencies):
   ```bash
   pip install torch==1.13.1+cpu torchvision==0.15.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
   ```

4. **Install remaining packages**:
   ```bash
   pip install streamlit pandas transformers
   ```

5. **Run the app**:
   ```bash
   streamlit run huggingface_app.py
   ```
