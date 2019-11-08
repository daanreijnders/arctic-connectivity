#/bin/bash

# SGE: the job name
#$ -N ParticleAdvectionDT60
#$ -V
#$ -l h_rt=72:00:00 
#$ -l hostname=science-bs36 #make sure it is in the right node
#
#
# SGE: your Email here, for job notification
#$ -M b.j.h.r.reijnders@uu.nl
#
# SGE: when do you want to be notified (b : begin, e : end, s : error)?
#$ -m e 
#$ -m s
#
# SGE: ouput in the current working dir
#$ -wd /science/users/4302001/arctic-connectivity/advection

source $HOME/start_conda.sh
cd /science/users/4302001/arctic-connectivity/advection

# COMMANDS HERE
python3 ../tools/advectParticles.py 20010101
