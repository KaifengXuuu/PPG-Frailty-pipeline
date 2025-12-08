# PPG_Frailty_pipeline
Extract Features from PPG signal for supporting classifying frailty situation of elder people


## Quick Start

### 1. Install Conda

If you don’t have Conda yet, install **Miniconda** or **Anaconda** and verify:

```bash
conda --version
conda create -n <a name you like> python=3.9
conda activate <a name you like>
git clone https://github.com/<your-username>/PPG-Frailty-pipeline.git
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




