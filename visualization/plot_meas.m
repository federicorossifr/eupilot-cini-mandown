function [mean_val, std_val] = plot_meas(measure_data,mname,task_title,fname, legends)
    frame_count = 1:1:size(measure_data,1);

    plot(frame_count,measure_data);
    legend(legends)
    title(task_title);
    xlabel("Frame number");
    ylabel(mname);
    saveas(gca,strcat("plots/",fname));

    boxplot(measure_data);
    xticklabels(legends);
    title(task_title);
    ylabel(mname)
    saveas(gca,strcat("plots/mean_",fname))

    close all;
    mean_val = mean(measure_data);
    std_val = std(measure_data);
end

