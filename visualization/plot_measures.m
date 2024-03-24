%% import
measures = import_measures("info.csv");

%% Task speed




pre_process_fps = 1000./measures.pre_process_speed;
inference_fps = 1000./measures.inference_speed;
post_process_fps = 1000./measures.post_process_speed;
man_down_fps = 1000./measures.man_down_speed;
deep_sort_fps = 1000./measures.deep_sort_speed;

total_frame_time = measures.pre_process_speed+measures.inference_speed+measures.post_process_speed+measures.man_down_speed+measures.deep_sort_speed;
total_fps = 1000./total_frame_time;

%% Plot FPS 

plot_fps(measures.pre_process_speed,"Pre-process task speed","preproc.png");
plot_fps(measures.inference_speed,"YOLOv5 task speed","yolov5.png");
plot_fps(measures.post_process_speed,"Post-process task speed","postproc.png");
plot_fps(measures.man_down_speed,"Man-down task speed","mandown.png");
plot_fps(measures.deep_sort_speed,"Deep-sort task speed","deepsort.png");
plot_fps(total_frame_time,"Overall task speed","total.png");


%% Plot measures

plot_meas(measures.CPU_utilization_rate,"Utilization (%)","CPU Utilization","cpu_util.png");
plot_meas(measures.CPU_temperature,"Temperature (°C)","CPU Temperature","cpu_temp.png");
plot_meas(measures.GPU_memory_used,"Memory (MB)","GPU Memory use","gpu_mem.png");
plot_meas(measures.GPU_utilization_rate,"Utilization (%)","GPU Utilization","gpu_util.png");
plot_meas(measures.GPU_temperature,"Temperature (°C)","GPU Temperature","gpu_temp.png");
plot_meas(measures.GPU_power_consumption,"Power (W)","GPU Instant power draw","gpu_draw.png");
