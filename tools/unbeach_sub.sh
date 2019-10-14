#/bin/sh

# SGE: the job name
#$ -N Antibeach
#$ -V
#$ -l h_rt=24:00:00
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
#$ -wd /home/students/4302001/arctic-connectivity/tools

source $HOME/start_conda.sh
cd /home/students/4302001/arctic-connectivity/tools

# COMMANDS HERE
python3 unbeach_Afield.py