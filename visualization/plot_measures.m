%% import
measures_gh200 = import_measures("raw_data/info_gh200.csv");
measures_armn1 = import_measures("raw_data/info_arm_n1.csv");
measures_a100 = import_measures("raw_data/info_a100.csv");
measures_jtx = import_measures("raw_data/info_jetson.csv");
measures_t4 = import_measures("raw_data/info_t4.csv"); %

%% Task stats

gh200_stats=compute_stats(measures_gh200);

%% Combine frametimes and measures

pre_process_speed = [measures_gh200.pre_process_speed measures_armn1.pre_process_speed measures_a100.pre_process_speed measures_jtx.pre_process_speed measures_t4.pre_process_speed ];
inference_speed = [measures_gh200.inference_speed measures_armn1.inference_speed measures_a100.inference_speed measures_jtx.inference_speed measures_t4.inference_speed];
post_process_speed = [measures_gh200.post_process_speed measures_armn1.post_process_speed measures_a100.post_process_speed measures_jtx.post_process_speed measures_t4.post_process_speed];
man_down_speed = [measures_gh200.man_down_speed measures_armn1.man_down_speed measures_a100.man_down_speed measures_jtx.man_down_speed measures_t4.man_down_speed];
deep_sort_speed = [measures_gh200.deep_sort_speed measures_armn1.deep_sort_speed measures_a100.deep_sort_speed measures_jtx.deep_sort_speed measures_t4.deep_sort_speed];
total_frame_time = [measures_gh200.total_frame_time measures_armn1.total_frame_time measures_a100.total_frame_time measures_jtx.total_frame_time measures_t4.total_frame_time];

CPU_utilization_rate = [measures_gh200.CPU_utilization_rate measures_armn1.CPU_utilization_rate measures_a100.CPU_utilization_rate measures_jtx.CPU_utilization_rate measures_t4.CPU_utilization_rate];
CPU_temperature = [measures_gh200.CPU_temperature measures_armn1.CPU_temperature measures_a100.CPU_temperature measures_jtx.CPU_temperature measures_t4.CPU_temperature];
GPU_memory_used = [measures_gh200.GPU_memory_used measures_armn1.GPU_memory_used measures_a100.GPU_memory_used measures_jtx.GPU_memory_used measures_t4.GPU_memory_used];
GPU_utilization_rate = [measures_gh200.GPU_utilization_rate measures_armn1.GPU_utilization_rate measures_a100.GPU_utilization_rate measures_jtx.GPU_utilization_rate measures_t4.GPU_utilization_rate];
GPU_temperature = [measures_gh200.GPU_temperature measures_armn1.GPU_temperature measures_a100.GPU_temperature measures_jtx.GPU_temperature measures_t4.GPU_temperature];
GPU_power_consumption = [measures_gh200.GPU_power_consumption measures_armn1.GPU_power_consumption measures_a100.GPU_power_consumption measures_jtx.GPU_power_consumption./1000 measures_t4.GPU_power_consumption];

legend_labels = {"NVIDIA GH200","ARM N1","NVIDIA A100","JETSON ORIN", "NVIDIA TESLA T4"};



%% Plot FPS for all architectures 


plot_fps(pre_process_speed,"Pre-process task speed","preproc.png",legend_labels);
plot_fps(inference_speed,"YOLOv5 task speed","yolov5.png",legend_labels);
plot_fps(post_process_speed,"Post-process task speed","postproc.png",legend_labels);
plot_fps(man_down_speed,"Man-down task speed","mandown.png",legend_labels);
plot_fps(deep_sort_speed,"Deep-sort task speed","deepsort.png",legend_labels);
plot_fps(total_frame_time,"Overall task speed","total.png",legend_labels);

%% Plot measures

plot_meas(CPU_utilization_rate,"Utilization (%)","CPU Utilization","cpu_util.png",legend_labels);
plot_meas(CPU_temperature,"Temperature (°C)","CPU Temperature","cpu_temp.png",legend_labels);
plot_meas(GPU_memory_used,"Memory (MB)","GPU Memory use","gpu_mem.png",legend_labels);
plot_meas(GPU_utilization_rate,"Utilization (%)","GPU Utilization","gpu_util.png",legend_labels);
plot_meas(GPU_temperature,"Temperature (°C)","GPU Temperature","gpu_temp.png",legend_labels);
plot_meas(GPU_power_consumption,"Power (W)","GPU Instant power draw","gpu_draw.png",legend_labels);
