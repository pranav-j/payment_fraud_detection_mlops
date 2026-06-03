# syntax=docker/dockerfile:1.7
FROM public.ecr.aws/lambda/python:3.12-arm64

COPY lambda/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY lambda/handler.py ${LAMBDA_TASK_ROOT}/handler.py
COPY feature_repo/ ${LAMBDA_TASK_ROOT}/feature_repo/

ENV PYTHONPATH="${LAMBDA_TASK_ROOT}/src:${PYTHONPATH}"
CMD ["handler.handler"]
