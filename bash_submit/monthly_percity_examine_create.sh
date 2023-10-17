create=0
if [ ${create} -eq 1 ]
then
    for i in {0..84}
    do
	cat monthly_fit_examine_template.sh | sed s/REPLACE/${i}/ > monthly_fit_examine_${i}.sh
	cat ../monthly_fit_examine.py | sed s/REPLACE/${i}/ > ../monthly_fit_examine_${i}.py
    done
fi


clean=1
if [ ${clean} -eq 1 ]
then
    for i in {0..84}
    do
	rm monthly_fit_examine_${i}.sh
	rm ../monthly_fit_examine_${i}.py
    done
fi


submit=0
if [ ${submit} -eq 1 ]
then
    for i in {0..84}
    do
	sbatch monthly_fit_examine_${i}.sh
    done
fi
