# Assessing Ocean Surface Connectivity in the Arctic: Capabilities and caveats of community detection in Lagrangian Flow Networks

This repository contains Python scripts and notebooks used in my master's research project on investigating connectivity in the surface of the Arctic Ocean. Author is Daan Reijnders unless specified otherwise.

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

## Thesis abstract
Community detection algorithms from the field of network theory have been used to divide a fluid domain into clusters that are sparsely connected with each other and to identify barriers to transport, for example in the context of larval dispersal. Communities detected by the community detection algorithm *Infomap* have barriers that have been shown to often coincide with well-known oceanographic features. Thus far, this method has only been applied to closed domains such as the Mediterranean. We apply this method to the surface of the Arctic and subarctic oceans and show that it can be applied to open domains. First, we construct a Lagrangian flow network by simulating the exchange of Lagrangian particles between different bins in an icosahedral-hexagonal grid. Then, *Infomap* is applied to identify groups of well-connected bins. The resolved transport barriers include naturally occurring structures, such as the major currents. As expected, clusters in the Arctic are affected by seasonal and decadal variations in sea-ice concentration. We also discuss several caveats of this method. Firstly, there is no single definition of what makes a cluster, since this is dependent on a preferred balance of internally high connectivity, sparse connectivity between clusters, and the spatial scale of investigation. Secondly, many different divisions into clusters may qualify as good solutions and it may thus be misleading to only consider the solution that optimizes a certain quality parameter the most. Finally, while certain cluster boundaries lie consistently at the same location between different good solutions, other boundary locations vary significantly, making it difficult to assess the physical meaning of a single solution. Particularly in the context of practical applications like planning Marine Protected Areas, it is important to consider an ensemble of qualifying solutions to find persistent boundaries.
