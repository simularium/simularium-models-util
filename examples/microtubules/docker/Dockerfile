FROM continuumio/miniconda3

RUN mkdir working

COPY env.yml .
RUN conda env create -f env.yml
SHELL ["conda", "run", "-n", "simularium_models", "/bin/bash", "-c"]

COPY src/ .
COPY microtubules.sh /microtubules.sh
RUN chmod +x /microtubules.sh

ENTRYPOINT /microtubules.sh