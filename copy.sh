#!/bin/bash
set -e

src="../../ml_course_bak/"
dest="./"

files=(lib/font/kg-second-chances-sketch/7de4544cc3e616ff945907bf07e2f271.eot \
       lib/font/kg-second-chances-sketch/7de4544cc3e616ff945907bf07e2f271.svg \
       lib/font/kg-second-chances-sketch/7de4544cc3e616ff945907bf07e2f271.ttf \
       lib/font/kg-second-chances-sketch/7de4544cc3e616ff945907bf07e2f271.woff \
       lib/font/kg-second-chances-sketch/7de4544cc3e616ff945907bf07e2f271.woff2 \
       css/style.css \
       datasets/datasets_generation/titanic_manifest.ipynb \
       datasets/datasets_generation/nyc_taxi_rides.ipynb \
       datasets/titanic_manifest.csv \
       datasets/nyc_taxi_rides.csv \
       media/black-board.jpg \
       media/diagrams/diagrams.xml \
       media/diagrams/titanic_process.png \
       media/diagrams/workflow_intro.png \
       media/diagrams/workflow_density_estimation.png \
       media/diagrams/random_process.png \
       html/workshop_01_intro.html \
       html/workshop_02_non_parametric_density_estimation.html \
       postBuild \
       requirements.txt \
       widgets/dice_desity_estimation.html \
       widgets/fit_normal.html \
       widgets/select_hist_bins.html \
       workshops/workshop_01_intro.ipynb \
       workshops/workshop_02_non_parametric_density_estimation.ipynb \
       )
       # README.md \

for file in "${files[@]}"
do
    cp $src$file $dest$file
done
