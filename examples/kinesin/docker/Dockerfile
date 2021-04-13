FROM continuumio/miniconda3

RUN mkdir working

COPY env.yml .
RUN conda env create -f env.yml
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

COPY src/ .
COPY kinesin.sh /kinesin.sh
RUN chmod +x /kinesin.sh

ENTRYPOINT /kinesin.sh