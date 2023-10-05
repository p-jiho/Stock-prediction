# Paraller 폴더 속의 모든 파일을 순차적으로 병렬처리 시도
import papermill as pm
from glob import glob

for nb in glob('./paraller/*.ipynb'):
    print(nb)

for nb in glob('./paraller/*.ipynb'):
    pm.execute_notebook(
        input_path=nb,
        output_path=nb,
        engine_name='embedded',
    )