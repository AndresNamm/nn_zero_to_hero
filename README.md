It is assume that in root folder you have runnable code. 
The references within modules have been built to reference from repo root. 


If for some reason you need to run some code within some folder, its better you set PYTHONPATH to repo root because there the default PYTHONPATH will be from the file location when running code. This can cause problems for imports. 