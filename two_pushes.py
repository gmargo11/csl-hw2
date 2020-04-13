import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from push_env import PushingEnv

from forward_model import ForwardModel
from inverse_model import InverseModel



if __name__ == "__main__":
    # train inverse model
    inverse_model = InverseModel()
    num_epochs=20
    train_losses, valid_losses = inverse_model.train(num_epochs=num_epochs)

    #train forward model
    forward_model = ForwardModel()
    num_epochs=20
    train_losses, valid_losses = forward_model.train(num_epochs=num_epochs)

    env = PushingEnv(ifRender=False)
    num_trials=10

    # two pushes, inverse model
    errors = np.zeros(num_trials)
    # save one push
    errors[0] = env.plan_inverse_model_extrapolate(inverse_model, img_save_name="inverse_twopush", seed=0)
    print("test loss:", errors[0])
    # try 10 random seeds
    for seed in range(1,10):
        errors[seed] = env.plan_inverse_model_extrapolate(inverse_model, seed=seed)
        print("test loss:", errors[seed])
    
    print("average loss, inverse model:", np.mean(errors))

    # two pushes, forward model
    errors = np.zeros(num_trials)
    # save one push
    errors[0] = env.plan_inverse_model_extrapolate(forward_model, img_save_name="forward_twopush", seed=0)
    print("test loss:", errors[0])
    # try 10 random seeds
    for seed in range(1,10):
        errors[seed] = env.plan_inverse_model_extrapolate(forward_model, seed=seed)
        print("test loss:", errors[seed])
    
    print("average loss, forward model:", np.mean(errors))