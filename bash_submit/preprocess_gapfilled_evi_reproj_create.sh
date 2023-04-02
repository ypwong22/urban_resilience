create=0
if [ ${create} -eq 1 ]
then
    for i in {2001..2019}
    do
	for j in 0 1
	do
	    cat preprocess_gapfilled_evi_reproj_template.sh | sed s/REPLACE1/${i}/g | sed s/REPLACE2/${j}/g > preprocess_gapfilled_evi_reproj_${i}_${j}.sh
	    cat ../preprocess_gapfilled_evi_reproj.py | sed s/REPLACE1/${i}/g | sed s/REPLACE2/${j}/g > ../preprocess_gapfilled_evi_reproj_${i}_${j}.py
	done
    done
fi


clean=1
if [ ${clean} -eq 1 ]
then
    for i in {2001..2019}
    do
	for j in 0 1
	do
	    rm preprocess_gapfilled_evi_reproj_${i}_${j}.sh
	    rm ../preprocess_gapfilled_evi_reproj_${i}_${j}.py
	done
    done
fi


submit=0
if [ ${submit} -eq 1 ]
then
    for i in {2001..2019}
    do
	for j in 0 1
	do
	    sbatch preprocess_gapfilled_evi_reproj_${i}_${j}.sh
	done
    done
fi
