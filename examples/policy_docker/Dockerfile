FROM python:3.12-slim

WORKDIR /app
RUN python -m pip install --no-cache-dir \
    grpcio==1.85.3 protobuf==7.36.2 numpy==2.3.1

COPY robofinals/__init__.py /app/robofinals/__init__.py
COPY robofinals/distributed/chamber_schema.py /app/robofinals/distributed/chamber_schema.py
COPY robofinals/distributed/grpc_policy.py /app/robofinals/distributed/grpc_policy.py
COPY robofinals/distributed/policy_v1_pb2.py /app/robofinals/distributed/policy_v1_pb2.py
COPY examples/policy_docker/example_policy.py /app/example_policy.py

ENTRYPOINT ["python", "-u", "/app/example_policy.py"]
