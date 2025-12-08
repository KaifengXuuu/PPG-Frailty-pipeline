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
Change the path of your ppg datasets folder in [ ] of the code below in Notebook ./PPG_Analy_Visual_test.ipynb.
```
envi = 1

windows_address_1 = ["/mnt/d/Tubcloud/Shared/PPG/Test Data",
                   "/mnt/d/Tubcloud/Shared/PPG/Test Data/25July25"]
```




