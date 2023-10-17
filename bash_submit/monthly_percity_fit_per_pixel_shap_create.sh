create=0
if [ ${create} -eq 1 ]
then
    for o in daymet topowx yyz
    do
    for i in heat_wave # hot_and_dry
    do
	for j in DJF MAM JJA SON
	do
	for k in {0..4}
	do
	for m in Resistance Recovery
	do
	for n in Sign Mag
	do
		cat monthly_percity_fit_per_pixel_shap_template.sh | sed s/REPLACE0/${o}/g | sed s/REPLACE2/${i}/g | sed s/REPLACE3/${j}/g | sed s/REPLACE4/${k}/g | sed s/REPLACE5/${m}/g | sed s/REPLACE6/${n}/g > monthly_percity_fit_per_pixel_shap_${o}_${i}_${j}_${k}_${m}_${n}.sh
		cat ../monthly_percity_fit_per_pixel.py | sed s/REPLACE0/${o}/g | sed s/REPLACE1/shap/g | sed s/REPLACE2/${i}/g | sed s/REPLACE3/${j}/g | sed s/REPLACE4/${k}/g | sed s/REPLACE5/${m}/g | sed s/REPLACE6/${n}/g > ../monthly_percity_fit_per_pixel_shap_${o}_${i}_${j}_${k}_${m}_${n}.py
	done
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
	for k in {0..4}
	do
	for m in Resistance Recovery
	do
	for n in Sign Mag
	do
		rm monthly_percity_fit_per_pixel_shap_${o}_${i}_${j}_${k}_${m}_${n}.sh
		rm ../monthly_percity_fit_per_pixel_shap_${o}_${i}_${j}_${k}_${m}_${n}.py
	done
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
	for k in {0..4}
	do
	for m in Recovery # Resistance Recovery
	do
	for n in Mag # Sign
	do
	    sbatch monthly_percity_fit_per_pixel_shap_${o}_${i}_${j}_${k}_${m}_${n}.sh
	done
	done
	done
	done
	done
	done
fi
