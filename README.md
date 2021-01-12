# Ocean Surface Connectivity in the Arctic: Capabilities and Caveats of Community Detection in Lagrangian Flow Networks

This repository contains Python scripts and notebooks used in my MSc research project on investigating connectivity in the surface of the Arctic Ocean. Author is Daan Reijnders unless specified otherwise.

Publication (open access):
Reijnders, D., van Leeuwen, E. J., & van Sebille, E. (2021). Ocean surface connectivity in the Arctic: Capabilities and caveats of community detection in Lagrangian flow networks. *Journal of Geophysical Research: Oceans*, 126, e2020JC016416. https://doi.org/10.1029/2020JC016416

## Data used
Hydrodynamical data from the `GLOBAL_REANALYSIS_PHY_001_030` dataset from the Copernicus Marine Environment Monitoring Service (CMEMS) is used. Data is loaded above 60N. Data is used between 1993 and 2018.

## Environment
Packages used can be found in the import list of each file. The Conda environment used in this thesis can be found in the `meta` directory. 

## Main pipeline
1. Particles are initialized with the `community.particles` class.
2. A `parcels.fieldset` is initialized with `fieldsetter_cmems` script.
3. Particles are advected using [Parcels](https://github.com/OceanParcels/parcels) through the `advectParticles` script.
4. A Lagrangian flow network is constructed using the `community.countBins` and `community.transMat` classes.
5. The community detection algorithm *Infomap* (version 1.0.0-beta.51) is applied on the Lagrangian flow network (`.net` file). Options used:
    * `-d` specifies that the network is directed
    * `-k` include self-edges
    * `--clu` print a .clu file with the top cluster ids for each node
    * `--markov-time` to specify the markov-time (almost always 2)
    * `-N 20` always run Infomap 20 times and choose the best solution
    * `-s` random seed. Almost always `314159`, unless seeking degenerate solutions. Then the range of seeds is 1..100.
6. The resulting community description (`.clu` file) is loaded using `community.countBins`.
7. Results are plotted and further analyzed in notebooks in `community_detection` directory.

Transition matrices (`npz`), corresponding network descriptions (`.net`) and community divisions (`.clu`) are persistenly stored at https://public.yoda.uu.nl/science/UU01/IN0OU9.html.

## Research Abstract
To identify barriers to transport in a fluid domain, community detection algorithms from network science have been used to divide the domain into clusters that are sparsely connected with each other. In a previous application to the closed domain of the Mediterranean Sea, communities detected by the _Infomap_ algorithm have barriers that often coincide with well-known oceanographic features. We apply this clustering method to the surface of the Arctic and subarctic oceans and thereby show that it can also be applied to open domains. First, we construct a Lagrangian flow network by simulating the exchange of Lagrangian particles between different bins in an icosahedral-hexagonal grid. Then, _Infomap_ is applied to identify groups of well-connected bins. The resolved transport barriers include naturally occurring structures, such as the major currents. As expected, clusters in the Arctic are affected by seasonal and annual variations in sea-ice concentration. An important caveat of community detection algorithms is that many different divisions into clusters may qualify as good solutions. Moreover, while certain cluster boundaries lie consistently at the same location between different good solutions, other boundary locations vary significantly, making it difficult to assess the physical meaning of a single solution. We therefore consider an ensemble of solutions to find persistent boundaries, trends and correlations with surface velocities and sea-ice cover.
