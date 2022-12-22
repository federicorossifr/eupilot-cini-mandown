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
            self.GPU_memory_used.append(s[5])
            self.GPU_utilization_rate.append(s[6])
            self.mem_utilization_rate.append(s[7])
            self.GPU_temperature.append(float(s[8]))
            self.GPU_power_consumption.append(float(s[9]))

        # Define number of frames:
        self.x = []
        for i in range(1, self.N + 1):
            self.x.append(i)
    
    def average_pre_process_speed(self):
        # Compute averafe pre-process speed in milliseconds (ms)
        N = self.N
        sum = np.sum(self.pre_process_speed)
        average = round(np.divide(sum, N), 1)

        return average
    def pre_process_speed_plot(self):

        x = self.x
        y = self.pre_process_speed
        y[0] = self.average_pre_process_speed()
        y[1] = y[0]
        plt.plot(x, y)
        plt.grid(visible = True)
        plt.title('Pre-Process Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N)
        plt.ylim()
        plt.show()

    def average_inference_speed(self):
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
        plt.plot(x, y)
        plt.grid()
        plt.title('Inference Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N)
        plt.show()

    def average_nms_speed(self):
        # Compute average inference speed in milliseconds (ms)
        N = self.N
        sum = np.sum(self.post_process_speed)
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

    def deep_sort_speed_plot(self):
        x = self.x
        y = self.deep_sort_speed
        # y[0] = self.average_post_process_speed()
        # y[1] = y[0]
        plt.plot(x, y)
        plt.grid()
        plt.title('Deep Sort Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.show()

    def GPU_temperature_plot(self):
        
        x = self.x
        y = self.GPU_temperature
        plt.plot(x, y)
        plt.grid(linewidth = 0.5)
        plt.title('GPU Temperature')
        plt.xlabel('Frame [N]')
        plt.ylabel('Temperature [°C]')
        plt.show()
    
    def average_GPU_temperature(self):
        # Compute average GPU Temperature in Celsius (°C)
        N = self.N
        sum = np.sum(self.GPU_temperature)
        average = round(np.divide(sum, N), 1)

        return average

    def average_GPU_power_consumption(self):
        # Compute average GPU Power Consumption in Watt (W)
        N = self.N
        sum = np.sum(self.GPU_power_consumption)
        average = round(np.divide(sum, N), 1)

        return average

    def GPU_power_consumption_plot(self):
        
        x = self.x
        y = self.GPU_power_consumption
        plt.plot(x, y)
        plt.grid(visible = True, linewidth = 0.5)
        plt.title('GPU Power Consumption')
        plt.xlabel('Frame [N]')
        plt.ylabel('Power [W]')
        plt.ylim(0, max(y) + 5)
        plt.show()