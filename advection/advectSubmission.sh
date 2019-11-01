#/bin/bash

# SGE: the job name
#$ -N ParticleGridAdvectionMesh
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
#$ -wd /home/students/4302001/arctic-connectivity/advection

source $HOME/start_conda.sh
cd /home/students/4302001/arctic-connectivity/advection

# COMMANDS HERE
#python3 ../tools/advectParticles.py rcp85 360 60 20000109 -d 30 -dt 20
#python3 ico_advection.py
python3 wedge_advection2051.py 20511201
