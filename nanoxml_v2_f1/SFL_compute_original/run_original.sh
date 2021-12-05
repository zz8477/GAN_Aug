#!/bin/sh

export subject_dir=nanoxml_v2_f1
export version=2

echo compute statements suspiciousness with SFL
./analysePro/sfl 0 result/componentinfo.txt result/error.txt result/covMatrix.txt 
./analysePro/sfl 2 result/componentinfo.txt result/error.txt result/covMatrix.txt
./analysePro/sfl 3 result/componentinfo.txt result/error.txt result/covMatrix.txt
./analysePro/sfl 4 result/componentinfo.txt result/error.txt result/covMatrix.txt
./analysePro/sfl 5 result/componentinfo.txt result/error.txt result/covMatrix.txt

echo moving SFL_result.txt to result
cp 0_Russel_Rao.txt result/v${version}
mv 0_Russel_Rao.txt result/v${version}slice
cp 2_GP03.txt result/v${version}
mv 2_GP03.txt result/v${version}slice
cp 3_GP19.txt result/v${version}
mv 3_GP19.txt result/v${version}slice
cp 4_OPTIMAL_P.txt result/v${version}
mv 4_OPTIMAL_P.txt result/v${version}slice
cp 5_GP191.txt result/v${version}
mv 5_GP191.txt result/v${version}slice

cd result
echo get final result
gcc -o translate translate.c
./translate 0_Russel_Rao v2/0_Russel_Rao.txt v2slice/0_Russel_Rao.txt
./translate 2_GP03 v2/2_GP03.txt v2slice/2_GP03.txt
./translate 3_GP19 v2/3_GP19.txt v2slice/3_GP19.txt
./translate 4_OPTIMAL_P v2/4_OPTIMAL_P.txt v2slice/4_OPTIMAL_P.txt
./translate 5_GP191 v2/5_GP191.txt v2slice/5_GP191.txt
rm -rf translate

cd ..

mv result/result.txt result.txt
