task_names=("sim_transfer_cube_decoupled_3cam_scripted_GEOM0.005_SITE0.005")
seeds=("2" "3")
num_episodes=("50")
batch_sizes=("8")
policy_classes=("Diffusion")

for seed in ${seeds[@]} ; do
    for task_name in ${task_names[@]} ; do
        for num_episode in ${num_episodes[@]} ; do
            for batch_size in ${batch_sizes[@]} ; do
                for policy_class in ${policy_classes[@]} ; do
                    sbatch run.sh ${task_name} ${seed} ${num_episode} ${batch_size} ${policy_class}
                done
            done
        done
    done
done

