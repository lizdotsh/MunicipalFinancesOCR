# Municipal Finances OCR

This page (and project) are still under development. Paused for summer 2022, work should continue in fall 2022.  

File Structure : 

+ MunicipalFinanceOCR
	+  rename_to_config.yml
		- Default yaml config file, rename to "config.yml" when using program 
	+  src/ 
		+ process.py
			- Primary file, collection of functions used in processing stage, parses results from OCR and det2 model into tabular csv format.
		+ det2.py
			- Functions used to run the Detectron2 model using the Tablebank dataset to identify the position of every table in the PDF
		+ muni.py 
			- WIP
		+ ocr.py 
			- Collection of functions dealing with Using Google's OCR engine to identify the position and content of every piece of text on each PDF. 

	 
