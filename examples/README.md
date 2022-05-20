# Develop locally

To set up your environment with a local editable install of `simularium_model_utils`:
```bash
cd [path to root of repo]
conda env create -f env.yml
conda activate myenv
pip install -e .[dev]
# or on mac
pip install -e .\[dev\]
```
Run example model, create outputs/ directory and logs:
```bash
cd [model]
python docker/src/[model].py template.xlsx 1 test
```

# Build Docker Image

To install docker:<br/>
`brew install --cask docker`<br/>
then open the Docker app in Applications/ and give it privileged access.

To build the docker image (named `readdy-actin` and tagged `v1.0`):<br/>
`cd [path to examples dir]/[model]/docker`<br/>
`docker build --no-cache -t readdy-actin:v1.0 ./`

To run a docker container (named `readdy-actin-test`) using the image LOCALLY:<br/>
`docker run --rm -v [path to examples dir]/[model]/:/working/ -e SIMULATION_TYPE='LOCAL' -e PARAM_SET_NAME='template' -e JOB_ARRAY_INDEX=0 --name readdy-actin-test readdy-actin:v1.0`

# Upload Image to AWS ECR

If this is the first time, install the AWS command line tools with `pip install awscli`, and then set up your access keys. Go to AWS IAM web console and download keys for your user as json. To set up your keys, use `aws configure`, which will guide you through a wizard.

In the AWS ECR web console, create a repository (in this example called `blairl/readdy-actin`)

Log in to ECR/Docker:<br/>`aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 108503617402.dkr.ecr.us-west-2.amazonaws.com`

Add the repo's 'latest' tag to docker image:<br/>
`docker tag readdy-actin:v1.0 108503617402.dkr.ecr.us-west-2.amazonaws.com/blairl/readdy-actin:latest`

Push the docker image to the repo in ECR:<br/>
`docker push 108503617402.dkr.ecr.us-west-2.amazonaws.com/blairl/readdy-actin:latest`

# Run Models on AWS Batch

Create a job definition in AWS Batch:<br/>
`aws batch register-job-definition --cli-input-json file://[path to examples dir]/[model]/[model]_job_definition.json`

Upload the parameters sheet (e.g. `template.xlsx`) to AWS S3 (or do this in the AWS S3 web console):<br/>
`aws s3 cp [path to examples dir]/[model]/template.xlsx s3://readdy-working-bucket/parameters/template.xlsx`

Go to the AWS Batch web console and submit a new job.

# Visualize Model output

If a model didn't output `.simularium` visualization files, you can generate them manually from the output `.h5` file(s). 

Download the `.h5` files from AWS S3 web console.

If you haven't already, create a conda environment with `conda env create -f [path to examples dir]/[model]/docker/env.yml`.

Activate the environment: `conda activate myenv`.

Use `cd [path to examples dir]/[model]` to navigate to the script directory.

Run the visualization script, e.g. `python visualize_[model].py [path to directory containing .h5 file(s)] [box size] [total steps]`