# Use a Python base image
FROM continuumio/miniconda3

# Setting the author name
LABEL author="BURLE HEMANTH"

# Copy the environment file
COPY ./deploy/conda/env.yml .

# Update the conda environment
RUN conda update conda

# Create the environment
RUN conda env create -f env.yml

# Activate the environment
RUN echo "source activate mle-dev" > ~/.bashrc

# Set environment variable
ENV PATH /opt/conda/envs/mle-dev/bin:$PATH

# Create working directory
RUN mkdir mle-training

# Expose port for MLflow UI
EXPOSE 5000

# Copy the project files
COPY . /mleProject

# Set working directory
WORKDIR /mleProject/

# Start the MLflow server
CMD mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5000

