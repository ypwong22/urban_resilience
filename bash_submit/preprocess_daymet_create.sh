create=0
if [ ${create} -eq 1 ]
then
    for i in prcp tmax tmin srad vp
    do
	for j in {1980..2020}
	do
	    cat preprocess_daymet_template.sh | sed s/REPLACE1/${i}/g | sed s/REPLACE2/${j}/g > preprocess_daymet_${i}_${j}.sh
	    cat ../preprocess_daymet.py | sed s/REPLACE1/${i}/g | sed s/REPLACE2/${j}/g > ../preprocess_daymet_${i}_${j}.py
	done
    done
fi


clean=1
if [ ${clean} -eq 1 ]
then
    for i in prcp tmax tmin srad vp
    do
	for j in {1980..2020}
	do
	    rm preprocess_daymet_${i}_${j}.sh
	    rm ../preprocess_daymet_${i}_${j}.py
	done
    done
fi


submit=0
if [ ${submit} -eq 1 ]
then
    for i in prcp tmax tmin srad vp
    do
	for j in {1980..2020}
	do
	    sbatch preprocess_daymet_${i}_${j}.sh
	done
    done
fi
