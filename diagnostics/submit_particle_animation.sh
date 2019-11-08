#/bin/sh

# SGE: the job name
#$ -N particle_animation
#$ -V
#$ -l hostname=science-bs35 #make sure it is in the right node
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
#$ -wd /science/users/4302001/arctic-connectivity/diagnostics

source $HOME/start_conda.sh
cd /science/users/4302001/arctic-connectivity/diagnostics

# COMMANDS HERE
python particle_animation.py