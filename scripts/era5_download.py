#!/usr/bin/env python3
"""Sanitized ERA5 download helper derived from TGRS.ipynb.
Set CDSAPI_KEY in the environment. Never commit the key.
"""
import argparse,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--start',type=int,default=1958); p.add_argument('--end',type=int,default=2023); a=p.parse_args()
    key=os.environ.get('CDSAPI_KEY')
    if not key: raise SystemExit('Set CDSAPI_KEY in the environment')
    Path.home().joinpath('.cdsapirc').write_text(f'url: https://cds.climate.copernicus.eu/api\nkey: {key}\n')
    import cdsapi
    variables=['2m_temperature','maximum_2m_temperature_since_previous_post_processing','minimum_2m_temperature_since_previous_post_processing','total_precipitation','potential_evaporation','surface_solar_radiation_downwards','surface_net_solar_radiation','10m_u_component_of_wind','10m_v_component_of_wind','2m_dewpoint_temperature']
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); client=cdsapi.Client()
    client.retrieve('reanalysis-era5-land-monthly-means',{'product_type':['monthly_averaged_reanalysis'],'variable':variables,'year':[str(y) for y in range(a.start,a.end+1)],'month':[f'{m:02d}' for m in range(1,13)],'time':['00:00'],'area':[42.5,95.5,32.0,120.0],'data_format':'netcdf','download_format':'unarchived'},str(out/f'ERA5_YRB_{a.start}_{a.end}.nc'))
if __name__=='__main__': main()
