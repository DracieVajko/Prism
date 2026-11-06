import sys
sys.path.insert(0, '.')
import download_models as dm
for t in dm.OCR_TASKS:
    print('ID:', t['id'], 'Repo:', t['repo'], 'Size:', t['min_gb'], 'GB')