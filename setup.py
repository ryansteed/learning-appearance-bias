from distutils.core import setup

setup(
    name='appearance_bias',
    version='1.0',
    packages=['appearance_bias',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    install_requires=[
    	'enlighten==1.1.0',
		'face_recognition==1.2.3',
		'imagenetscraper',
		'tensorflow-hub==0.1.1',
		'tensorflow',
		'tensorboard==1.12.0',
		'numpy>=1.15.4',
		'keras',
		'pandas',
		'scikit-learn',
		'grequests',
		'matplotlib',
		'seaborn',
		'ImageScraper',
		'Pillow',
		'jupyterlab',
		'lime',
		'ipywidgets',
		'shap'
    ]
)