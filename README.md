# Collocation-based Neural ODE Training

1. **Clone the Repository**
Clone the repository to your local machine using:
```bash
git clone https://github.com/mariashapo/NODE
cd NODE
```

2. **Set Up the Environment**
Create a new conda environment using the provided `environment.yml` file:
```bash
conda create -n node25_local python=3.9 pip -c conda-forge
conda activate node25_local
python -m pip install -e ".[viz,analysis,utils,pyomo,torch,torch-extras,jax-cpu,optim]"
```

Initialise packages
```bash
pip install -e .
```

3. **Running the synthetic experiments:**
Default Pyomo run, further parameters can be added as needed. Some other parameters are in the config file in `src/config_pyomo_synth.json`. 
```bash
python -m src.synthetic_data.training_conv_pyomo --layer_width '[2,32,2]' --exp 'default'
```

4. **Running real-life data experiments:**
Default Pyomo run for real-life data experiments. All parameters can be adjusted in the 'src.real_life_data.train_pyomo.py' file itself. 
```bash
python -m src.real_life_data.train_pyomo
```