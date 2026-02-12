# Base image
FROM python:3.11

# Install all required packages to run the model
RUN apt update && apt install --yes libgl1 libglib2.0-0
