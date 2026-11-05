FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /work/apps/listings_simulator

# Install the simulator's own dependencies.
COPY apps/listings_simulator/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Include both entry points and their helpers.
COPY apps/listings_simulator/simulator ./simulator
