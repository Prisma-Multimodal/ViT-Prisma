from setuptools import setup, find_packages


install_requires = [
    'torch',
    'torchvision',
    'numpy',
    'jaxtyping',
    'einops',
    'fancy_einsum',
    'timm',
    'transformers',
    'scikit-learn',
    'datasets',
    'line_profiler',
    'open-clip-torch',
]

visualization_requires = ['plotly>=6.1.1,<7', 'matplotlib', 'kaleido>=1,<2']
tracking_requires = ['wandb']

setup(
    name='vit-prisma',
    version='2.0.0',
    author='Sonia Joseph',
    author_email='soniamollyjoseph@gmail.com',
    description='A Vision Transformer library for mechanistic interpretability.',
    long_description=open('docs/README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/soniajoseph/vit-prisma',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={
    'vit_prisma': ['visualization/*.html', 'visualization/*.js'],
    # Add other patterns here as needed
},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.10',
    install_requires=install_requires,
    keywords='vision-transformer, clip, multimodal, machine-learning, mechanistic interpretability',
    zip_safe=False,
    extras_require={
        'test': ['pytest>=7'],
        'visualization': visualization_requires,
        'tracking': tracking_requires,
        'all': visualization_requires + tracking_requires,
        'sae': ['sae-lens==2.1.3'],
        'arrow': ['pyarrow']  # to use: pip install -e .[sae] # as of 2.1.3, windows will require pip install sae-lens==2.1.3 --no-dependencies followed by manually installing needed packages
    },
    entry_points={
        'console_scripts': [
            'prisma-cpu-quickstart=vit_prisma.examples.cpu_quickstart:main',
        ],
    },
)
