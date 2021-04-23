To install docker:<br/>
`brew install --cask docker`<br/>
then open the Docker app in Applications/ and give it privileged access.

To build the docker image (named `readdy-actin` and tagged `v1.0`):<br/>
`cd [path to this dir]/[model]/docker`<br/>
`docker build -t readdy-actin:v1.0 ./`

To run a docker container (named `readdy-actin-test`) using the image LOCALLY:<br/>
`docker run -v [path to this dir]/[model]/:/working/ -e SIMULATION_TYPE='LOCAL' -e PARAM_SET_NAME='template' -e JOB_ARRAY_INDEX=0 --name readdy-actin-test readdy-actin:v1.0`

To run on AWS Batch:<br/>
1. If this is the first time, install the AWS command line tools, use `pip install awscli`

2. If this is the first time, set up your access keys. Go to AWS IAM console and download keys for your user as json. To set up your keys, use `aws configure`

3. In the AWS ECR console, create a repository (in this example called `blairl/readdy-actin`)

4. Log in to ECR/Docker:<br/>`aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 108503617402.dkr.ecr.us-west-2.amazonaws.com`

5. Add the repo's 'latest' tag to docker image:<br/>
`docker tag readdy-actin:v1.0 108503617402.dkr.ecr.us-west-2.amazonaws.com/blairl/readdy-actin:latest`

6. Push the docker image to the repo in ECR:<br/>
`docker push 108503617402.dkr.ecr.us-west-2.amazonaws.com/blairl/readdy-actin:latest`

7. Create a job definition in AWS Batch:<br/>
`aws batch register-job-definition --cli-input-json file://[path to this dir]/[model]/[model]_job_definition.json`

8. Upload the parameters sheet (e.g. `template.xlsx`) to AWS S3 (or do this in the AWS S3 console):<br/>
`aws s3 cp [path to this dir]/[model]/template.xlsx s3://readdy-working-bucket/parameters/template.xlsx`

9. Go to the AWS Batch console and submit a new job.