# Security

A CDS API credential was embedded in the uploaded `TGRS.ipynb`. It has been removed from all notebooks and scripts in this release. The credential owner should revoke/rotate it immediately.

Use the environment variable `CDSAPI_KEY` with `scripts/era5_download.py`. Do not commit `.cdsapirc`, tokens, passwords, model-service credentials, or private dataset URLs.
