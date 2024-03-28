# Solving real‑world optimization tasks using physics‑informed neural computing
## Physics-informed neural networks (PINNs)


## Pendulum
The paper produces the following image, where the angle of the pendulum and the torque are plotted against time.
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/0a0f63d8-eaa0-4d7d-9b4a-3f0163460b6b)

Our first reproduction yielded the following results:

Once we removed the L-BFGS optimizer and allowed the Adam optimizer to run for a longer amount of iteration, the model converged after approximately 11,000 iterations.
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/fbd21380-4a54-4efc-a0c4-70a8f3c1fa4d)

In figure 5, the variation in the learning curve was shown. First, the goal loss was plotted for five random seeds. This yielded the following results:
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/bd4658ee-4823-43e4-8fb8-45bde09a2575)

Our reproduction also implemented 5 random seeds, which gave the following plot:

The same was done for the angle of the pendulum. Side by side, the figure produced by the original paper and the figure that we reproduced were as follows:
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/6141b447-facb-4e54-bc19-b814031e6025)




## Brachistone curve

## Fermat's principle

## Swingby

## Conclusion
