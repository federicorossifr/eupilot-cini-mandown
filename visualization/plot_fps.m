function [] = plot_fps(task_frame_time,task_title,fname)
    task_fps = 1000./task_frame_time;
    frame_count = 1:1:size(task_fps,1);

    plot(frame_count,task_fps);
    title(task_title);
    xlabel("Frame number");
    ylabel("Task speed (FPS)")
    saveas(gca,fname);
    close all;
end

