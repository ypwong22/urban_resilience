create=0
if [ ${create} -eq 1 ]
then
    for o in daymet topowx yyz
    do
    for i in heat_wave # hot_and_dry
    do
	for j in DJF MAM JJA SON
	do
	for k in {0..5}
	do
	for m in Resistance Recovery
	do
		cat monthly_percity_fit_per_pixel_regression_template.sh | sed s/REPLACE0/${o}/g | sed s/REPLACE2/${i}/g | sed s/REPLACE3/${j}/g | sed s/REPLACE4/${k}/g | sed s/REPLACE5/${m}/g > monthly_percity_fit_per_pixel_regression_${o}_${i}_${j}_${k}_${m}.sh
		cat ../monthly_percity_fit_per_pixel.py | sed s/REPLACE0/${o}/g | sed s/REPLACE1/regression/g | sed s/REPLACE2/${i}/g | sed s/REPLACE3/${j}/g | sed s/REPLACE4/${k}/g | sed s/REPLACE5/${m}/g > ../monthly_percity_fit_per_pixel_regression_${o}_${i}_${j}_${k}_${m}.py
	done
    done
	done
	done
	done
fi


clean=1
if [ ${clean} -eq 1 ]
then
    for o in daymet topowx yyz
    do
    for i in heat_wave # hot_and_dry
    do
	for j in DJF MAM JJA SON
	do
	for k in {0..5}
	do
	for m in Resistance Recovery
	do
		rm monthly_percity_fit_per_pixel_regression_${o}_${i}_${j}_${k}_${m}.sh
		rm ../monthly_percity_fit_per_pixel_regression_${o}_${i}_${j}_${k}_${m}.py
	done
    done
	done
	done
	done
fi


submit=0
if [ ${submit} -eq 1 ]
then
    for o in daymet topowx yyz
    do
    for i in heat_wave # hot_and_dry
    do
	for j in DJF MAM JJA SON
	do
	for k in {0..5}
	do
        for m in Recovery # Resistance Recovery
	do
	    sbatch monthly_percity_fit_per_pixel_regression_${o}_${i}_${j}_${k}_${m}.sh
	done
	done
	done
	done
	done
fi
