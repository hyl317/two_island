#!/bin/bash
#$ -S /bin/bash #defines bash as the shell for execution
#$ -N twoIsland_nfARG #Name of the command that will be listed in the queue
#$ -cwd #change to current directory
#$ -j y #join error and standard output in one file, no error file will be written
#$ -q archgen.q #queue
#$ -m e #send an email at the end of the job
#$ -M yilei_huang@eva.mpg.de #send email to this address
#$ -pe smp 12 #needs 8 CPU cores
#$ -l h_vmem=30G #request 4Gb of memory
#$ -V # load personal profile
#$ -o $JOB_NAME.o.$JOB_ID.$TASK_ID
#$ -t 1:21:1

id=$SGE_TASK_ID
id=$(($id-1))

Ts=(20 30 50)
Ns=(50 100 250 500 1000 2000 5000)

Nindex=$(($id/3))
N=${Ns[$Nindex]}
id=$(($id-3*$Nindex))
T=${Ts[$id]}

echo T$T
echo N$N
python3 sim2Island_notFullARG.py -N $N -T $T -r 50 -p 12 -e 1000