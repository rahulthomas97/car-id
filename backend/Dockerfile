FROM python:3.9
RUN pip install --no-cache-dir --timeout=120 fastapi uvicorn boto3 python-multipart
RUN pip install --no-cache-dir --timeout=120 torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Set the working directory in the container
WORKDIR /app
# Copy the application files
COPY app.py /app
# Create the serve script that SageMaker expects
RUN echo '#!/bin/bash\n\
uvicorn app:app --host 0.0.0.0 --port 8080' > /usr/local/bin/serve

# Make the serve script executable
RUN chmod +x /usr/local/bin/serve

# Expose the API port
EXPOSE 8080

# Set the entrypoint to the serve script
ENTRYPOINT ["serve"]
