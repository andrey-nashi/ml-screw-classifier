# ml-screw-classifier
Detection of defects in screws

**Requirements**

- PC with GPU
- Docker installed (with given sudo privileges)

--------------------------------------
**Install**

- Download all necessary data (images, masks, weights, etc)
```
cd tools
python3 run-download-data.py
```

If the script has issues with downloading due to permissions the following use the 
following g-drive links and extract in `./data`




- Build docker container (based on cuda 11.03, check driver compatibility)
```
cd docker
docker-compose build
```

**Replicating results**

Execute the shell script `run-replicate-results.sh`


**Training**

Execute the shell script `run-replicate-training.sh`

--------------------------------------
**Utils**

- Generate synthetic image, deffect mask pairs and save in the directory `tools/synthetic`

```
cd tools
python3 run-gen-synthetic.py
```

--------------------------------------

Note
- Docker may not run correctly if there is a cudnn/driver version mismatch. 
Make virtual environment (use requirements in dir `docker`) and run natively. 

Tested
* Natively i7-8700k, 1080ti
* Dockerized i5, rtx 3070 GN20-E5