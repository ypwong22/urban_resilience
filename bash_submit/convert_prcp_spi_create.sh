create=0
if [ ${create} -eq 1 ]
then
    for i in `seq 0 84`
    do
	cat convert_prcp_spi_template.sh | sed s/REPLACE/${i}/ > convert_prcp_spi_${i}.sh
	cat ../convert_prcp_spi.py | sed s/REPLACE/${i}/ > ../convert_prcp_spi_${i}.py
    done
fi


clean=1
if [ ${clean} -eq 1 ]
then
    for i in `seq 0 84`
    do
	rm convert_prcp_spi_${i}.sh
	rm ../convert_prcp_spi_${i}.py
    done
fi


submit=0
if [ ${submit} -eq 1 ]
then
    for i in `seq 0 84`
    do
	sbatch convert_prcp_spi_${i}.sh
    done
fi
