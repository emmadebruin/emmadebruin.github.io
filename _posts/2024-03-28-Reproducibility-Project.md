# Solving real‑world optimization tasks using physics‑informed neural computing
## Physics-informed neural networks (PINNs)
Traditional scientific computing often relies on two distinct approaches: physics-based models and data-driven learning. Physics-based models, built upon established physical laws, can accurately represent the underlying behavior of a system but can be computationally expensive and may require simplification for complex scenarios. Conversely, data-driven approaches such as machine learning excel at uncovering hidden patterns in data but often lack interpretability and struggle with extrapolation beyond the observed data. Physics-informed neural networks (PINNs) bridge this gap by offering a powerful framework that uses the strengths of both approaches. PINNs are neural networks that are trained not only on data but also on the governing physical laws that describe the system. In some instances, PINNs can work without or with very little training data. This allows them to learn complex dynamics from experimental measurements or simulations while remaining faithful to the underlying physical principles. This unique ability empowers PINNs to capture intricate phenomena that might be challenging to model with traditional methods. Additionally, PINNs offer the potential to reduce computational costs compared to high-fidelity simulations, especially for tasks that involve repeated evaluations or exploration of a large parameter space. Moreover, by incorporating both physics and data, PINNs have the potential to generalize better to unseen scenarios, making them valuable tools for scientific discovery and engineering applications.  In this blogpost, we reproduce the paper "Solving real‑world optimization tasks using physics‑informed neural computing" by Jaemin Seo [1]. In this paper, a number of real-world optimization tasks are addressed using PINNs. The goal of this blogpost is to determine whether this paper is reproducible and whether the results of the paper are supported by this reproducibility study.

## Pendulum
The first real-world optimization task which the paper addresses is that of a pendulum. A situation is sketched where a pendulum is attached to a 1 meter long string, to which a maximum torque of 2.5N in both directions can be applied. The aim of the system is to ensure that the pendulum reaches an inverted state after 10 seconds by applying varying amounts of torque in varying directions. This network uses three loss functions: the physical loss, the constraint loss and the goal loss. These are calculated as follows:

![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/35ff7552-a78b-4374-9b89-bd62f68d92f5)

![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/03dbf588-0bf6-44bb-9cf7-de0687bad8ce)

![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/3fca6c3c-5002-48ef-a2b7-776b2ce4268f)

In these loss functions, we incorporate the function F. This is the governing equation. PINNs use governing equations in order to incorporate physical laws into their loss functions in order to penalize outputs which do not follow the relevant laws of physics. In this case, F is the equation of motion under gravity.

![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/16c9d969-fad7-4119-8256-f0e7e0cb040e)

Each of these are weighted in order to form a total loss. The weights of the physical and goal loss are 1, whereas the weight of the constraint loss is 10. These are hyperparameters set by the author of the paper. The paper produces the image on the left, where the angle of the pendulum and the torque are plotted against time, and this can be compared to our reproduction on the right.

![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/0a0f63d8-eaa0-4d7d-9b4a-3f0163460b6b {width=40px height=400px})
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/d500c284-07fa-4018-a9e4-35029068aa28)

This shows that the model does not converge when using Adam for 5000 epochs and subsequently using L-BFGS. There was insufficient information in the paper in order to adjust the L-BFGS optimizer so that it would work in our reproduction. The decision was therefore made to remove the L-BFGS optimizer and continue using Adam until convergence. Once the L-BFGS optimizer was removed and the Adam optimizer was allowed to run for a longer amount of iterations, the model converged after approximately 11,000 iterations. This is slightly longer than the time it took in the original paper, but this is attributed to their addition use of the L-BFGS optimizer.

![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/fbd21380-4a54-4efc-a0c4-70a8f3c1fa4d)

In figure 5, the variation in the learning curve was shown. First, the goal loss was plotted for five random seeds. Below, the results of the paper can be seen on the left and the results of our reproduction can be seen on the right:

![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/bd4658ee-4823-43e4-8fb8-45bde09a2575)
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/b774f0d2-9ba9-44a3-ae21-98441cdce504)

The shape of this plot matches the shape of the plot from the original paper. However, it is important to note that the goal loss is reduced much faster and more efficiently in the original paper compared to ours.


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

### References
[1] J. Seo, “Solving real-world optimization tasks using physics-informed neural computing,” Scientific Reports, vol. 14, no. 1, Jan. 2024, doi: 10.1038/s41598-023-49977-3.
