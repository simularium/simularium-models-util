To install docker:<br/>
`brew install --cask docker`<br/>
then open the Docker app in Applications/ and give it permissions.

To build the docker image:<br/>
`cd [path to this dir]/docker`<br/>
`docker build -t readdy-microtubules:v1.0 ./`

To run a docker container using the image LOCALLY:<br/>
`docker run -v [path to this dir]:/working/ -e SIMULATION_TYPE='LOCAL' -e PARAM_SET_NAME='template' -e JOB_ARRAY_INDEX=0 --name readdy-microtubules-test readdy-microtubules:v1.0`