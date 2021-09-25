FROM continuumio/miniconda3

RUN mkdir working

COPY env.yml .
RUN conda env create -f env.yml
SHELL ["conda", "run", "-n", "models", "/bin/bash", "-c"]

COPY src/ .
COPY actin.sh /actin.sh
RUN chmod +x /actin.sh

ENTRYPOINT /actin.sh