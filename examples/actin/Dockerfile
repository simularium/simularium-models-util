FROM continuumio/conda-ci-linux-64-python3.8

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN conda config --add channels conda-forge
RUN conda install -c readdy/label/dev readdy
RUN mkdir input output
COPY src/ .

RUN chmod +x /actin.sh
