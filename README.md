# Reliable Off-Policy Learning for Dosage Combinations
 Implementation of "Reliable Off-Policy Learning for Dosage Combinations".

 <p align="center">
  <img src="media/method.png?raw=true" width="600"/>
</p>

Install requirements:
```
pip install -r requirements.txt
```

To run the experiments on the TCGA dataset:

1. Download the TCGA dataset version from https://github.com/ioanabica/SCIGAN and store it in "dataset/tcga/".
2. Specify the parameters and model options in "conf/config.yaml".
3. Run the run.sh script with the following command:
```
bash run.sh
```
The results are then stored in "logs/" in an own subfolder per each setting.