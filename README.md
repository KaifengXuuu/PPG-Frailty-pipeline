# PPG_Frailty_pipeline
Extract Features from PPG signal for supporting classifying frailty situation of elder people


## Quick Start

### 1. Install Conda

If you don’t have Conda yet, install **Miniconda** or **Anaconda** and verify:

```bash
conda --version
conda create -n PPG-frality python=3.9.0
conda activate PPG-frality
git clone https://github.com/KaifengXuuu/PPG-Frailty-pipeline.git
cd PPG-Frailty-pipeline
pip install .
```

### 2. Change filepath
1. Open the folder /PPG-Frailty-pipeline
2. Right click in space of folder and select "Terminal"
3. Run following commands:
```
git fetch -a
git switch main
git pull
cp .env.example .env
code .
```
4. Open the file /.env in VSCode and change the folowing variables to the Path of your PPG-Datasets:
```
folderpath1 = "/home/Test Data"
folderpath2 = "/home/Test Data/25July25" 
```
5. Open file /PPG_Analy_Visual_test.ipynb and click "Run all" butten in VSCode.




