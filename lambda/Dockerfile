# syntax=docker/dockerfile:1.7
# AWS Lambda Python 3.12 base image (arm64)
FROM public.ecr.aws/lambda/python:3.12-arm64

# Install dependencies
COPY lambda/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project source so fraud_mlops.inference is importable
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY lambda/handler.py ${LAMBDA_TASK_ROOT}/handler.py

# Add src to Python path
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}/src:${PYTHONPATH}"

# Lambda handler entrypoint
CMD ["handler.handler"]
