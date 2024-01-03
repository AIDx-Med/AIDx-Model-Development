# Use Hugging Face's Transformers PyTorch image as the base
FROM huggingface/transformers-pytorch-gpu

# Set the working directory in the container
WORKDIR /workspace

# Install Packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888


COPY start-jupyter.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start-jupyter.sh

# Command to run Jupyter Notebook
CMD ["start-jupyter.sh"]
