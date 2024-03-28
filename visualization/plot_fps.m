function [mean_val, std_val] = plot_fps(task_frame_times,task_title,fname,legends)
    task_fps = 1000./task_frame_times;
    frame_count = 1:1:size(task_fps,1);

    plot(frame_count,task_fps);
    legend(legends)
    title(task_title);
    xlabel("Frame number");
    ylabel("Task speed (FPS)")
    saveas(gca,strcat("plots/",fname));

    boxplot(task_fps);
    xticklabels(legends);
    title(task_title);
    ylabel("Task speed (FPS)")
    saveas(gca,strcat("plots/mean_",fname))

    close all;
    mean_val = mean(task_fps);
    std_val = std(task_fps);
end

