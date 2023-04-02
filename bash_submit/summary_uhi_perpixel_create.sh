create=0
if [ ${create} -eq 1 ]
then
    for i in {0..84}
    do
	cat summary_uhi_perpixel_template.sh | sed s/REPLACE/${i}/ > summary_uhi_perpixel_${i}.sh
	cat ../summary_uhi_perpixel.py | sed s/REPLACE/${i}/ > ../summary_uhi_perpixel_${i}.py
    done
fi


clean=1
if [ ${clean} -eq 1 ]
then
    for i in {0..84}
    do
	rm summary_uhi_perpixel_${i}.sh
	rm ../summary_uhi_perpixel_${i}.py
    done
fi


submit=0
if [ ${submit} -eq 1 ]
then
    for i in {0..84}
    do
	sbatch summary_uhi_perpixel_${i}.sh
    done
fi
