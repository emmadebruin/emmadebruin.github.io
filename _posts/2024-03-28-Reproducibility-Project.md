# Solving real‑world optimization tasks using physics‑informed neural computing
## Physics-informed neural networks (PINNs)
Traditional scientific computing often relies on two distinct approaches: physics-based models and data-driven learning. Physics-based models, built upon established physical laws, can accurately represent the underlying behavior of a system but can be computationally expensive and may require simplification for complex scenarios. Conversely, data-driven approaches such as machine learning excel at uncovering hidden patterns in data but often lack interpretability and struggle with extrapolation beyond the observed data. Physics-informed neural networks (PINNs) bridge this gap by offering a powerful framework that uses the strengths of both approaches. PINNs are neural networks that are trained not only on data but also on the governing physical laws that describe the system. In some instances, PINNs can work without or with very little training data. This allows them to learn complex dynamics from experimental measurements or simulations while remaining faithful to the underlying physical principles. This unique ability empowers PINNs to capture intricate phenomena that might be challenging to model with traditional methods. Additionally, PINNs offer the potential to reduce computational costs compared to high-fidelity simulations, especially for tasks that involve repeated evaluations or exploration of a large parameter space. Moreover, by incorporating both physics and data, PINNs have the potential to generalize better to unseen scenarios, making them valuable tools for scientific discovery and engineering applications.  In this blogpost, we reproduce the paper "Solving real‑world optimization tasks using physics‑informed neural computing" by Jaemin Seo [1]. In this paper, a number of real-world optimization tasks are addressed using PINNs. The goal of this blogpost is to determine whether this paper is reproducible and whether the results of the paper are supported by this reproducibility study.

## Pendulum
The first real-world optimization task which the paper addresses is that of a pendulum. A situation is sketched where a pendulum of mass 1 kg is attached to a 1 meter long string, to which a maximum torque (𝜏) of 2.5N in both directions can be applied. The aim of the system is to ensure that the pendulum reaches an inverted state after 10 seconds by applying varying amounts of torque in varying directions. The angle of the pendulum (ɸ) is important, as this can give the system information about the position of the pendulum, but also the speed and acceleration if it is differentiated with respect to time. The angle is measured in radians. 

The author of this paper has built a PINN which can efficiently solve this problem. The network in question contains three hidden layers of 64 neurons each, with a tanh activation function at each layer. At the output layer, another tanh activation function is applied to the torque, but not to the angle of the pendulum. This network uses three loss functions: the physical loss, the constraint loss and the goal loss. These are calculated as follows:

  <img src= "https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/35ff7552-a78b-4374-9b89-bd62f68d92f5" height="50"> <br/>
  <img src= "https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/03dbf588-0bf6-44bb-9cf7-de0687bad8ce" height="50"> <br/>
  <img src= "https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/3fca6c3c-5002-48ef-a2b7-776b2ce4268f" height="30"> <br/>

In physical loss function, we incorporate the function F. This is the governing equation. PINNs use governing equations in order to incorporate physical laws into their loss functions in order to penalize outputs which do not follow the relevant laws of physics. In this case, F is the equation of motion under gravity.

<center>
  <img src= "https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/16c9d969-fad7-4119-8256-f0e7e0cb040e" height="30"> <br/>
</center>
  
The constraint loss incorporates the boundary conditions of the problem and takes the initial conditions into account. These initial conditions are that the position and the angular velocity of the pendulum at time t=0 are 0. The goal loss tells the network that the position of the pendulum should be -1 at time t=10, in order to ensure that after 10 seconds, the pendulum is inverted.

Each of these losses are then weighted in order to form a total loss. The weights of the physical and goal loss are 1, whereas the weight of the constraint loss is 10. These are hyperparameters set by the author of the paper. Now the architecture of the network and the elements of the loss function are clear, the optimization process starts. This paper applies Adam optimization for 5000 epochs with a learning rate of 0.02. After this, the author makes the unusual decision to switch to a different optimizer. L-BFGS optimization is carried out until convergence of the system. The paper produces the image on the left, where the angle of the pendulum and the torque are plotted against time, and this can be compared to our reproduction on the right.

<img src="https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/0a0f63d8-eaa0-4d7d-9b4a-3f0163460b6b" width="350">
<img src="https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/d500c284-07fa-4018-a9e4-35029068aa28" width="650">

This shows that in our reproduction, the model does not converge when using Adam for 5000 epochs and subsequently using L-BFGS. We saw that Adam successfully lowered the loss until it stagnated around a total loss of 3, at which point L-BFGS optimization took over and caused the loss to increase to 80 and stay at this position. There was insufficient information in the paper in order to adjust the L-BFGS optimizer so that it would work in our reproduction. The decision was therefore made to remove the L-BFGS optimizer and continue using Adam until convergence. Once the L-BFGS optimizer was removed and the Adam optimizer was allowed to run for an increased number of iterations, the model converged after approximately 11,000 iterations. This is slightly longer than the amount of iterations it took in the original paper, which was said to be 9804 iterations, but this is attributed to their additional use of the L-BFGS optimizer.

<center>
  <img src= "https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/fbd21380-4a54-4efc-a0c4-70a8f3c1fa4d" width="350"> <br/>
</center>

In figure 5, the variation in the learning curve was shown. This analysis was done in order to assess the stability of the network. First, the goal loss was plotted for five random seeds. Below, the results of the paper can be seen on the left and the results of our reproduction can be seen on the right. The shape of the reproduced plot matches the shape of the plot from the original paper. However, it is important to note that the goal loss is reduced much faster and more efficiently in the original paper compared to ours. This is strange, since the goal loss in the original paper is reduced almost to 0 within 5000 learning steps, which implies that Adam optimization alone should be able to reduce this loss. Our reproduction did not use the L-BFGS optimizer, but this optimizer is only applied after the first 5000 learning steps. Therefore, this discrepancy between our reproduction and the original paper cannot be due to the L-BFGS optimizer.

<img src="https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/bd4658ee-4823-43e4-8fb8-45bde09a2575" width="350">
<img src="https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/b774f0d2-9ba9-44a3-ae21-98441cdce504" width="650"><br/>

The same was done for the angle of the pendulum. Five random seeds were used and the model was run once with each seed. Side by side, the figure produced by the original paper and the figure that we reproduced were as shown below. It is remarkable that while the two figures look alike, there is a significant difference. In the original paper, some of the seeds lead to large positive or negative angles, and do not converge. In our reproduction, the angles seem to remain within the inner dotted red lines drawn on the figure from the paper.

<img src="https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/6141b447-facb-4e54-bc19-b814031e6025" width="350">
<img src="https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/1b87d5a6-9074-47fa-b6a7-0c1617e7199f" width="650">


## Brachistone curve and Fermat's principle
The figure that can be seen in the original paper can be viewed below.
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/f64f045b-f964-4a17-a007-55b21ab0fbd7)
![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/d956d071-b40a-40c2-80b9-10bf4779c99f)

A physics informed neural network (PINN) is programmed to solve the problem of finding the shortest-time path connecting two given points were an analytical solution exits. For this, two experiments were conducted: Light path of the shortest time according to Fermat’s principle and the shortest time path under gravity according to the Brachistochrone curve. 
The 2D xy space contain a medium with a varying refractive index which has a sinusoidal profile with respect to the y-axis. Fermat’s principle states that light travels slower in a medium that contains a higher refractive index and that light travels between two points along the path that requires the least time [2]. In our experiment the refractive index is highest in the middle of the y-plane which would result in the light taking a more vertical direction. To find the shortest-time path using PINN the following governing equation is given by [1]

![image](/images/flight.png)

In case of the bird, the 2D xy space consists of a space where normal gravity applies and the bird can gain speed with gravitational energy only. Thus to get as quickly from the initial point (x0, y0) as (0, 1) to (1, 0) the bird is expected to follow brachistochrone path, which is not the shortest path distance wise but is the shortest path considering time as the bird is able to optimise the gaining of speed and the travelled distance. To find the shortest-time path that the bird should take using PINN the following governing equation is given by [1]

![image](/images/fbird.png)

These governing equations are then used in the following equation to calculate the physical loss to minimise for each experiment

![image](/images/lbird_phys.png)

The boundary loss is calculated using Dirichlet boundary conditions, these imply a certain penalty for errors at given coordinates in the solution. For the two experiments these consist solely of the start and destination coordinates, so that the boundary loss aims to ensure a correct initial and final position. This is implemented using the following formula, where BC represents the boundary condition coordinates and u corresponds to either the initial (x, y) or the final (x, y)

![image](/images/lbird_bc.png)

The goal loss, which is used to find the solution that has the lowest loss in terms of the defined goal is given as

![image](/images/lbird_goal.png)

where T is the total time to reach the destination point

All losses are then combined in the following equation, which aims to minimise the total combined loss, the weights are equal for both the experiments and are given as (1, 1, 0.01), respectively.

![image](/images/lbird_argmin.png)

To determine the shortest-time path by using PINN, the initial conditions for the network are set. The time is normalized between 0 and 1, with an appropriate uniform grid such that a 1000 time points are present in the given domain. We created a small function which can add more points on the boundary points to aid in model training, were we ensured that more training points near the boundary still had all nonzero time positional derivatives.

Additionally, the seed for each library is initialised at 0, to ensure that results are not dependent on a specific run. The constraints are set as the coordinates of the starting and ending points, respectively (0, 0), (1, 1) for the light path and respectively (0, 1),(1, 0) for the bird. An Adam optimizer was used with a 2000 epochs and a learning rate of 0.001. 

To create the PINN we created a class, Net, in python which extends the pytorch torch.nn.Module class. In our network we created an input layer which converts the single time input to 64 points. Then we applied 3 hidden layers, each of dimension 64x64. After each layer, a tanh activation function. Then the output layer converts the 64 neurons to the x, y coordinates and are then put into a final sigmoid activation function.

In our network we used the torch Xavier normal initialisation to initialise all our layers in the neural network.

Initially in the epoch, the optimizer is set to zero and a forward pass is executed. Our network calculates the loss which is used in the backward pass subsequentially. The loss is saved to give an indication on whether the neural network is converging to a solution. 

After having trained the PINN neural network, it is trained with a L-BFGS optimizer for 400 epochs, as in accordance with the original paper. We noted however that this yielded an insignificant change in loss or even none at all. We therefore think that this step is redundant and can be excluded in the case of these two experiments.

Using our trained PINN to generate the timepoints a 0 to 1 scale with a grid of a 100 total points yields the following two figures. What can be observed is that the bird shortest path deviates significantly form the analytical solution and does also not reach the final coordinate appropriately. The experiment with the shortest light path time yields an extremely similar final result when compared to the analytical solution. As we noted this difference, we rewrote our code to have a shared code base such that only the governing equation and boundary conditions differ, we then also ensured that all matrix computations gave the expected outputs and dimensions, and after correcting several issues, this still yielded the final plots as used in this blog. We have therefore concluded that we were unable to replicate the results for the bird.

![image](/images/bird.png) ![image](/images/light.png)

![image](/images/bird_loss.png) ![image](/images/light_loss.png)


## Swingby
The final real-world optimization task addressed in the paper concerns a Swingby trajectory. The scenario outlines a situation where a "rocket" with a mass of 1 kg traverses three celestial bodies with varying positions and masses. The objective of the system is to minimize the additional thrust required to propel the rocket. Understanding the forces acting upon the rocket is crucial, as they provide insight into the amount of thrust needed.

The author of this paper has developed a Physics-Informed Neural Network (PINN) capable of efficiently solving this problem. The network architecture comprises three hidden layers, each consisting of 64 neurons employing a tanh activation function. Another tanh activation function is applied at the output layer. This network employs two loss functions: the physical loss and the constraint loss, computed using the provided formulas.

In the case of the swingby trajectory, the physical loss function,  F is the equation for a thrust vector under gravitational foces: 

![image](https://github.com/stevengdemunck/docs/blob/main/assets/css/Screenshot%202024-04-08%20at%2022.50.59.png)

In our case, the constraint loss incorporates the boundary conditions of the problem and takes the initial conditions into account. These initial conditions are that the position starts at (x0,y0) = (-1,-1) at t=0 are and ends at (x1,y1) = (1,1).

Each of these losses is then assigned weights to formulate a total loss. The weights for the physical and constraint losses are both set to 1, serving as hyperparameters determined by the author. Initially, Adam optimization is applied for 2000 epochs with a learning rate of 0.001. The paper presents the image below, showing the path taken and plotting different gravitational forces and thrust against time.

![image](https://github.com/emmadebruin/emmadebruin.github.io/assets/165269949/d6ac0116-7ca8-4aa6-bddb-98c913afc2dd)

Similar to the pendulum reproduction scenario, the combination of L-BFGS and Adam proved ineffective. However, by adjusting the learning rate to 3*10^-3 and extending the number of epochs to 60000, we managed to replicate similar results.

This first image shows the optimal path that should be taken to minimze thrust.

![image](https://github.com/stevengdemunck/docs/blob/main/assets/css/Figure_1.png)

This image shows the total gravitational forces over time. When the rocket is closest to a celestial body, it experiences higher gravitational forces. 

![image](https://github.com/stevengdemunck/docs/blob/main/assets/css/gravity.png)

Here, the the total amount of force required to take the path over time is plotted

![image](https://github.com/stevengdemunck/docs/blob/main/assets/css/Figure_6.png)

Finally the total amount of added thrust is plotted over time. The values we obtained are higher when compared to the grapgh shown in the paper, this could be due to the fact that we did not implement the L-BGFS. 

![image](https://github.com/stevengdemunck/docs/blob/main/assets/css/Figure_7.png)

## Conclusion

### References
[1] J. Seo, “Solving real-world optimization tasks using physics-informed neural computing,” Scientific Reports, vol. 14, no. 1, Jan. 2024, doi: 10.1038/s41598-023-49977-3.

[2] S. Y. Mak, “A closer look at Fermat’s principle,” Phys. Educ., vol. 21, no. 6, p. 365, Nov. 1986, doi: 10.1088/0031-9120/21/6/005.
