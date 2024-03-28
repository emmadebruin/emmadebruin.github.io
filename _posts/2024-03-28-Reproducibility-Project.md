# Solving real‑world optimization tasks using physics‑informed neural computing
## Physics-informed neural networks (PINNs)


## Pendulum
The paper produces the following image, where the angle of the pendulum and the torque are plotted against time.
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/0a0f63d8-eaa0-4d7d-9b4a-3f0163460b6b)

Our first reproduction yielded the following results:

This shows that the model does not converge when using Adam for 5000 epochs and subsequently using L-BFGS. There was insufficient information in the paper in order to adjust the L-BFGS optimizer so that it would work in our reproduction. The decision was therefore made to remove the L-BFGS optimizer and continue using Adam until convergence. Once the L-BFGS optimizer was removed and the Adam optimizer was allowed to run for a longer amount of iterations, the model converged after approximately 11,000 iterations. This is slightly longer than the time it took in the original paper, but this is sttributed to their addition use of the L-BFGS optimizer.
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/fbd21380-4a54-4efc-a0c4-70a8f3c1fa4d)

In figure 5, the variation in the learning curve was shown. First, the goal loss was plotted for five random seeds. This yielded the following results:
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/bd4658ee-4823-43e4-8fb8-45bde09a2575)

Our reproduction also implemented 5 random seeds, which gave the following plot:
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/b774f0d2-9ba9-44a3-ae21-98441cdce504)


The same was done for the angle of the pendulum. Side by side, the figure produced by the original paper and the figure that we reproduced were as follows:
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/6141b447-facb-4e54-bc19-b814031e6025)
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/1b87d5a6-9074-47fa-b6a7-0c1617e7199f)





## Brachistone curve
The figure that can be seen in the original paper can be viewed below.
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/f64f045b-f964-4a17-a007-55b21ab0fbd7)


## Fermat's principle
The figure in the original paper that we aimed to reproduce can be seen below.
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/d956d071-b40a-40c2-80b9-10bf4779c99f)


## Swingby
The author of this paper produced the following figures for this problem:
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/d6ac0116-7ca8-4aa6-bddb-98c913afc2dd)


## Conclusion
