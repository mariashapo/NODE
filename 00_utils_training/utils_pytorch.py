import numpy as np

def prepare_custom_weights(wb_trained):
    """
    Extracts and prepares custom weights and biases from the provided dictionary,
    ensuring compatibility with PyTorch by converting from JAX arrays if necessary.
    
    Args:
    data_dict (dict): Dictionary containing layer parameters.
    
    Returns:
    list of tuples: Each tuple contains (weights, biases) tensors for a layer.
    """
    custom_params = {
        'Dense_0': {
            'kernel': np.array(wb_trained['W1']).T,
            'bias': np.array(wb_trained['b1'])
        },
        'Dense_1': {
            'kernel': np.array(wb_trained['W2']).T,
            'bias': np.array(wb_trained['b2'])
        }
    }
        
    custom_weights = []
    
    for key, params in custom_params.items():
        if 'Dense' in key:
            kernel = params['kernel']
            bias = params['bias']
            
            # Check if kernel and bias are JAX arrays, convert to numpy if true
            if hasattr(kernel, 'block_until_ready'):
                kernel = np.array(kernel)
            if hasattr(bias, 'block_until_ready'):
                bias = np.array(bias)
            
            custom_weights.append((kernel.T, bias.T))
    
    return custom_weights