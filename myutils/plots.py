'''Plots Utils'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class Plots:

    def __init__(self, file_path):

        # Define variables:
        self.pre_process_speed = []
        self.inference_speed = []
        self.nms_speed = []
        self.man_down_speed = []
        self.deep_sort_speed = []
        self.total_speed = []

        self.GPU_memory_used = []
        self.GPU_utilization_rate = []
        self.mem_utilization_rate = []
        self.GPU_temperature = []
        self.GPU_power_consumption = []

        # Read data from file:
        with open(file_path, 'r') as f:
            data = f.readlines()
            self.N = len(data)

        for s in data:
            s = s.replace('\n', '')
            s = s.split(' ')
            self.pre_process_speed.append(round(float(s[0]), 1))
            self.inference_speed.append(round(float(s[1]), 1))
            self.nms_speed.append(round(float(s[2]), 1))
            self.man_down_speed.append(round(float(s[3]), 1))
            self.deep_sort_speed.append(round(float(s[4]), 1))
            self.total_speed.append(round(float(s[5]), 1))
            self.GPU_memory_used.append(s[6])
            self.GPU_utilization_rate.append(s[7])
            self.mem_utilization_rate.append(s[8])
            self.GPU_temperature.append(float(s[9]))
            self.GPU_power_consumption.append(float(s[10]))

        # Define number of frames:
        self.x = []
        for i in range(1, self.N + 1):
            self.x.append(i)
    
    ### PRE-PROCESS SPEED ###
    def pre_process_average_speed(self):
        # Compute average pre-process speed in milliseconds (ms)
        N = self.N
        sum = np.sum(self.pre_process_speed)
        average = round(np.divide(sum, N), 1)

        return average
    def pre_process_speed_plot(self):

        x = self.x
        y = self.pre_process_speed
        # y[0] = self.pre_process_average_speed()
        # y[1] = y[0]
        plt.figure(num = 1)
        plt.plot(x, y)
        plt.grid(visible = True)
        plt.title('Pre-Process Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N)
        plt.ylim()
        plt.show()

    ### INFERENCE SPEED ###

    def inference_average_speed(self):
        # Compute average inference speed in milliseconds (ms)
        N = self.N
        sum = np.sum(self.inference_speed)
        average = round(np.divide(sum, N), 1)

        return average

    def inference_speed_plot(self):

        x = self.x
        y = self.inference_speed
        # y[0] = self.average_inference_speed()
        # y[1] = y[0]
        plt.figure(num = 2)
        plt.plot(x, y)
        plt.grid(visible = True)
        plt.title('Inference Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N)
        plt.ylim(0, max(y) + 100)
        plt.show()

    ### NON-MAXIMUM SUPPRESSION SPEED ###

    def nms_average_speed(self):
        # Compute average inference speed in milliseconds (ms)
        N = self.N
        sum = np.sum(self.nms_speed)
        average = round(np.divide(sum, N), 1)

        return average

    def nms_speed_plot(self):

        x = self.x
        y = self.nms_speed
        # y[0] = self.average_post_process_speed()
        # y[1] = y[0]
        plt.plot(x, y)
        plt.grid()
        plt.title('NMS Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.show()

    ### MANDOWN CLASSIFIER SPEED ###

    def man_down_average_speed(self):
        # Compute average inference speed in milliseconds (ms)
        N = self.N
        sum = np.sum(self.man_down_speed)
        average = round(np.divide(sum, N), 1)

        return average

    def man_down_speed_plot(self):
        x = self.x
        y = self.man_down_speed
        # y[0] = self.average_post_process_speed()
        # y[1] = y[0]
        plt.plot(x, y)
        plt.grid()
        plt.title('Man Down Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.show()

    ### DEEP SORT ALGORITHM SPEED ###

    def deep_sort_average_speed(self):
        # Compute average inference speed in milliseconds (ms)
        N = self.N
        sum = np.sum(self.deep_sort_speed)
        average = round(np.divide(sum, N), 1)

        return average

    def deep_sort_speed_plot(self):
        x = self.x
        y = self.deep_sort_speed
        # y[0] = self.average_post_process_speed()
        # y[1] = y[0]
        plt.plot(x, y)
        plt.grid()
        plt.title('DeepSORT Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.show()

    ### ALGORITHM TOTAL SPEED ###

    def total_average_speed(self):
        # Compute average inference speed in milliseconds (ms)
        N = self.N
        sum = np.sum(self.total_speed)
        average = round(np.divide(sum, N), 1)

        return average

    def total_speed_plot(self):

        x = self.x
        y = self.pre_process_speed + self.inference_speed + self.nms_speed + self.man_down_speed + self.deep_sort_speed
        plt.plot(x, y)
        plt.grid()
        plt.title('Total Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.show()

    ### TOTAL ###

    def speed_plot(self):

        plt.figure(num = 1)
        plt.plot(self.x, self.total_speed, label = 'Total')
        plt.plot(self.x, self.deep_sort_speed, label = 'DeepSORT')
        plt.plot(self.x, self.man_down_speed, label = 'Man Down Classifier')
        plt.plot(self.x, self.nms_speed, label = 'Post-Process')
        plt.plot(self.x, self.inference_speed, label = 'YOLO Inference')
        plt.plot(self.x, self.pre_process_speed, label = 'Pre-Process')
        plt.grid()
        plt.title('Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N)
        plt.ylim(-10, max(self.total_speed) + 30)
        plt.legend()
        plt.savefig('Speed')

    ### GPU TEMPERATURE ###

    def GPU_temperature_plot(self):

        sum = np.sum(self.GPU_temperature)
        average = round(np.divide(sum, self.N), 1)
        
        x = self.x
        y = self.GPU_temperature

        plt.figure(num = 2)
        plt.plot(x, y)
        plt.grid(linewidth = 0.5)
        plt.title('GPU Temperature')
        plt.xlabel('Frame [N]')
        plt.ylabel('Temperature [°C]')
        plt.xlim(0, self.N)
        plt.ylim(0, max(y) + 20)
        plt.savefig('GPU Temperature')

    ### GPU POWER CONSUMPTION

    def GPU_power_consumption_plot(self):
        
        average = []
        for i in range(0, self.N):
            sum = np.sum(self.GPU_power_consumption[0:i+1])
            average.append(round(np.divide(sum, i+1), 1))

        x = self.x
        y = self.GPU_power_consumption
        mean = np.round(np.sum(self.GPU_power_consumption)/self.N, 1)

        plt.figure(num = 3)
        plt.plot(x, y)
        plt.plot(x, average, label = f"""mean ({mean} W)""")
        plt.grid(visible = True, linewidth = 0.5)
        plt.title('GPU Power Consumption')
        plt.xlabel('Frame [N]')
        plt.ylabel('Power [W]')
        plt.xlim(0, self.N)
        plt.ylim(0, max(y) + 20)
        plt.legend(loc = 'lower left')
        plt.savefig('GPU Power Consumption')