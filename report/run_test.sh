CWD="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
output_dir=$CWD/output/
data_dir=$CWD/../data/
out_temp=${output_dir}temp.txt
test_image_base=squares
test_image_ext=.png
do_serial=${1}          # 0 for no serial computation 1 for yes do serial computation
out_file=${2}           # out.txt default

pushd ..
make

if [ "$out_file" == '' ]
then
    out_file=out.txt
fi

out_file=${output_dir}${out_file}

all_tests=(8 16 32 64 128 256 512 1024 2048 4096 8192)
if [ ! -d "${output_dir}" ]
then 
    echo Creating directory ${output_dir}
    mkdir ${output_dir}
fi
printf "n\t\tgpu_time\t\tcpu_time\n" >> ${out_temp}
for i in "${all_tests[@]}"
do
    name=test_${i}.txt
#    echo $i >> out_cpu.txt
    cpu_out=''
    if [ "$do_serial" == 1 ]
    then
        cpu_out=$(./main ${data_dir}${test_image_base}${test_image_ext} ${output_dir}${test_image_base}_cpu_${i}_${test_image_ext} ${i} ${i} 1)
    fi
    gpu_out=$(./main ${data_dir}${test_image_base}${test_image_ext} ${output_dir}${test_image_base}_gpu_${i}_${test_image_ext} ${i} ${i} 0)
    line="${i}\t\t${gpu_out}\t\t${cpu_out}\n"
    printf $line >> ${out_temp}
done
column -t ${out_temp} > ${out_file}
rm ${out_temp}
