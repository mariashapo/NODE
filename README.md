# Collocation-based Neural ODE Training

This project develops and tests a collocation-based methodology for efficiently training Neural Ordinary Differential Equations (Neural ODEs). 
By leveraging spectral methods and incorporating interpolation techniques, this framework aims to enhance the speed and accuracy of Neural ODE training processes. 

## Key Features

- **Collocation-Based Training:** Utilizes collocation methods to integrate constraints directly within the training process, enhancing the stability and convergence speed.
- **Hybrid Training Framework:** Combines traditional and collocation-based techniques to optimize model training and performance, especially on complex tasks.
- **Spectral Methods Integration:** Explores the use of spectral methods beyond traditional collocation approaches, potentially offering more robust and scalable solutions.
- **Real-World Application:** Tests the methodology on real-life data, assessing its effectiveness in practical scenarios.

## Key Files
```bash
NODE/
│
├── README.md          
├── environment.yml                     # The environment file for Python projects
├── 00_models/         
│   ├── nn_jax_diffrax.py               # Diffrax-JAX benchmark model
│   ├── nn_pytorch.py                   # Pytorch benchmark model
│   ├── nn_pyomo_base.py                # Collocation-based Pyomo model
│   ├── nn_pyomo_admm.py                # Collocation-based ADMM Pyomo model
│   └── ode_solver_pyomo_base.py        # Direct Collocation ODE solver
│   
├── 00_utils/            
│   ├── collocation_obj.py              # Collocation: Differential matrix and grid computation
│   ├── data_genration.py               # Data generation for synthetic data
│   ├── preprocessing.py                # Data preprocessing for real-world data
│   └── non_parametric_collocation.py   # Least 

```