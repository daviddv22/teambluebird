"""
Number of epochs. If you experiment with more complex networks you
might need to increase this. Likewise if you add regularization that
slows training.
"""
num_epochs = 1000

"""
A critical parameter that can dramatically affect whether training
succeeds or fails. The value for this depends significantly on which
optimizer is used. Refer to the default learning rate parameter
"""
learning_rate = .0002

"""
Beta_1 is the first hyperparameter for the Adam optimizer.
"""
beta_1 = .99

"""
epsilon for the Adam optimizer.
"""
epsilon = 1e-1

"""
A critical parameter for style transfer. The value for this will determine 
how much the generated image is "influenced" by the CONTENT image.
"""
alpha = .05

"""
A critical parameter for style transfer. The value for this will determine 
how much the generated image is "influenced" by the STYLE image.
"""
beta = 5