function [] = plot_meas(measure_data,mname,task_title,fname)
    frame_count = 1:1:size(measure_data,1);

    plot(frame_count,measure_data);
    title(task_title);
    xlabel("Frame number");
    ylabel(mname)
    saveas(gca,fname);
    close all;
end

