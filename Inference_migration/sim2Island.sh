#!/bin/bash
#$ -S /bin/bash #defines bash as the shell for execution
#$ -N twoIsland #Name of the command that will be listed in the queue
#$ -cwd #change to current directory
#$ -j y #join error and standard output in one file, no error file will be written
#$ -q archgen.q #queue
#$ -m e #send an email at the end of the job
# -M yilei_huang@eva.mpg.de #send email to this address
#$ -pe smp 8 #needs 8 CPU cores
#$ -l h_vmem=30G #request 4Gb of memory
#$ -V # load personal profile
#$ -o $JOB_NAME.o.$JOB_ID.$TASK_ID
#$ -t 1:30:1

id=$SGE_TASK_ID
id=$(($id-1))

T=20
Ms=(0.001 0.005 0.01 0.02 0.05 0.1)
Ns=(50 100 250 500 1000 2000)

Nindex=$(($id/6))
N=${Ns[$Nindex]}
id=$(($id-6*$Nindex))
m=${Ms[$id]}

echo T$T
echo N$N
echo m$m
python3 sim2Island.py -N $N -T $T -m $m -r 50 -p 12 -e 1000
