# MO-AMP Designer

### How to run it on your own machine

1. # Install the python version don't newer then v3.13

    https://www.python.org/downloads/

2. # active virtual environment run on powershell

   ```
   $ cd {to where you download path}
   $ python -m venv env
   ```
   
   ```
   $ py -{python version} -m venv env   # if you more then one version of python 
   ```
   ```
   $ Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
   $ .\amp_env\Scripts\Activate.ps1    #
   ```

3. # Install the requirements

   ```
   $ pip install wheel setuptools # for the first time
   $ pip install -r requirements.txt
   ```

4. # Run the app

   ```
   $ streamlit run CECI_app.py
   ```
