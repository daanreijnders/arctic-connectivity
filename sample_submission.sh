#/bin/sh

# SGE: the job name
#$ -N JOB NAME HERE
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
#$ -wd /home/students/4302001/arctic-connectivity/EEZ

source $HOME/start_conda.sh
cd /home/students/4302001/arctic-connectivity/EEZ

# COMMANDS HERE
