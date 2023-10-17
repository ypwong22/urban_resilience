create=0
if [ ${create} -eq 1 ]
then
    for i in {0..919}
    do
	cat preprocess_bnu_data_reproj_template.sh | sed s/REPLACE/${i}/g > preprocess_bnu_data_reproj_${i}.sh
	cat ../preprocess_bnu_data_reproj.py | sed s/REPLACE/${i}/g > ../preprocess_bnu_data_reproj_${i}.py
    done
fi


clean=1
if [ ${clean} -eq 1 ]
then
    for i in {0..919}
    do
	rm preprocess_bnu_data_reproj_${i}.sh
	rm ../preprocess_bnu_data_reproj_${i}.py
    done
fi


submit=0
if [ ${submit} -eq 1 ]
then
    for i in {0..919}
    do
	sbatch preprocess_bnu_data_reproj_${i}.sh
    done
fi
