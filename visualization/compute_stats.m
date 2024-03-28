function [stats_measures] = compute_stats(measures)
    stats_measures = struct;
    pre_process_fps = 1000./measures.pre_process_speed;
    inference_fps = 1000./measures.inference_speed;
    post_process_fps = 1000./measures.post_process_speed;
    man_down_fps = 1000./measures.man_down_speed;
    deep_sort_fps = 1000./measures.deep_sort_speed;
    total_frame_time = measures.pre_process_speed+measures.inference_speed+measures.post_process_speed+measures.man_down_speed+measures.deep_sort_speed;
    total_fps = 1000./total_frame_time;

    stats_measures.pre_process_fps = [mean(pre_process_fps), std(pre_process_fps)];
    stats_measures.inference_fps = [mean(inference_fps), std(inference_fps)];
    stats_measures.post_process_fps = [mean(post_process_fps), std(post_process_fps)];
    stats_measures.man_down_fps = [mean(man_down_fps), std(man_down_fps)];
    stats_measures.deep_sort_fps = [mean(deep_sort_fps), std(deep_sort_fps)];
    stats_measures.total_fps = [mean(total_fps), std(total_fps)];

    stats_measures.CPU_utilization_rate = [mean(measures.CPU_utilization_rate), std(measures.CPU_utilization_rate)];
    stats_measures.CPU_temperature = [mean(measures.CPU_temperature), std(measures.CPU_temperature)];
    stats_measures.GPU_memory_used = [mean(measures.GPU_memory_used), std(measures.GPU_memory_used)];
    stats_measures.GPU_utilization_rate = [mean(measures.GPU_utilization_rate), std(measures.GPU_utilization_rate)];
    stats_measures.GPU_temperature = [mean(measures.GPU_temperature), std(measures.GPU_temperature)];
    stats_measures.GPU_power_consumption = [mean(measures.GPU_power_consumption), std(measures.GPU_power_consumption)];
    
end

% plot_meas(measures_gh200.CPU_utilization_rate,"Utilization (%)","CPU Utilization","cpu_util.png");
% plot_meas(measures_gh200.CPU_temperature,"Temperature (°C)","CPU Temperature","cpu_temp.png");
% plot_meas(measures_gh200.GPU_memory_used,"Memory (MB)","GPU Memory use","gpu_mem.png");
% plot_meas(measures_gh200.GPU_utilization_rate,"Utilization (%)","GPU Utilization","gpu_util.png");
% plot_meas(measures_gh200.GPU_temperature,"Temperature (°C)","GPU Temperature","gpu_temp.png");
% plot_meas(measures_gh200.GPU_power_consumption,"Power (W)","GPU Instant power draw","gpu_draw.png");
