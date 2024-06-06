task_names=("aloha_thread_blue_needle_glue")
# task_names=("aloha_thread_blue_needle_glue")
seeds=("1")
batch_sizes=("8")
policy_classes=("ACT")
# policy_classes=("ACT")

for seed in ${seeds[@]} ; do
    for task_name in ${task_names[@]} ; do
        for batch_size in ${batch_sizes[@]} ; do
            for policy_class in ${policy_classes[@]} ; do
#                sbatch run.sh ${task_name} ${seed} ${batch_size} ${policy_class}
                 ./run.sh ${task_name} ${seed} ${batch_size} ${policy_class}
            done
        done
    done
done

