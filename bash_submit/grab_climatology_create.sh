create=1
if [ ${create} -eq 1 ]
then
    for i in daymet topowx yyz
    do
	for j in tiff tiff_3x
	do
	    cat grab_climatology_template.sh | sed s/REPLACE1/${i}/g | sed s/REPLACE2/${j}/g > grab_climatology_${i}_${j}.sh
	    cat ../grab_climatology.py | sed s/REPLACE1/${i}/g | sed s/REPLACE2/${j}/g > ../grab_climatology_${i}_${j}.py
	done
    done
fi


clean=0
if [ ${clean} -eq 1 ]
then
    for i in daymet topowx yyz
    do
	for j in tiff tiff_3x
	do
	    rm grab_climatology_${i}_${j}.sh
	    rm ../grab_climatology_${i}_${j}.py
	done
    done
fi


submit=1
if [ ${submit} -eq 1 ]
then
    for i in daymet topowx yyz
    do
	for j in tiff tiff_3x
	do
	    sbatch grab_climatology_${i}_${j}.sh
	done
    done
fi
