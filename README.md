# ml-screw-classifier
Detection of defects in screws

**Requirements**

- Machine with GPU
- Docker installed (with given sudo priveledges)


**Install**

- Download all necesary data 
- Build docker container
```
cd docker
docker-compose build
```

**Replicating results**

Execute the shell script `run-replicate-results.sh`


**Training**

Execute the shell script `run-replicate-training.sh`

**Utils**

- Generate synthetic image, deffect mask pairs and save in the directory `tools/synthetic`

```
cd tools
python3 run-gen-synthetic.py
```
