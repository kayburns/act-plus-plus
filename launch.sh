# task_names=("sim_tool_hang")
# task_names=("sim_transfer_cube_decoupled_3cam_scripted_GEOM0.005_SITE0.02")
# task_names=("aloha_fork_pass_in_cup")
task_names=("aloha_thread_blue_needle_glue")
seeds=("1")
# num_episodes=("200")
num_episodes=("50")

batch_sizes=("32")
policy_classes=("ACT")
# policy_classes=("Diffusion")

for seed in ${seeds[@]} ; do
    for task_name in ${task_names[@]} ; do
        for num_episode in ${num_episodes[@]} ; do
            for batch_size in ${batch_sizes[@]} ; do
                for policy_class in ${policy_classes[@]} ; do
                    sbatch run.sh ${task_name} ${seed} ${num_episode} ${batch_size} ${policy_class}
                    # ./run.sh ${task_name} ${seed} ${num_episode} ${batch_size} ${policy_class}
                done
            done
        done
    done
done

