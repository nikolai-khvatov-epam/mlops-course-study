FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Install dependencies (preserving minimal changes)
RUN pip install pandas pyarrow scikit-learn click

# Copy the script (uses image's model.bin)
COPY starter.py .

# Entrypoint for CLI params
ENTRYPOINT ["python", "starter.py"]