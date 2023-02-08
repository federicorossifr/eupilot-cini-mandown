
import numpy as np
import matplotlib.pyplot as plt

class ShowSpeed:

    def __init__(self, file_path):

        # Define variables:
        self.pre_process_speed = []
        self.inference_speed = []
        self.post_process_speed = []
        self.man_down_speed = []
        self.deep_sort_speed = []

        # Read data from file:
        with open(str(file_path + '/info.txt'), 'r') as f:
            data = f.readlines()
            self.N = len(data)

        for s in data:
            s = s.replace('\n', '')
            s = s.split(' ')
            self.pre_process_speed.append(round(float(s[0]), 1))
            self.inference_speed.append(round(float(s[1]), 1))
            self.post_process_speed.append(round(float(s[2]), 1))
            self.man_down_speed.append(round(float(s[3]), 1))
            self.deep_sort_speed.append(round(float(s[4]), 1))

        # Define number of frames:
        self.x = []
        for i in range(0, self.N):
            self.x.append(i)
    
    def pre_process_average_speed(self):
        # Compute average pre-process speed in milliseconds (ms)
        sum = np.sum(self.pre_process_speed)
        average = round(np.divide(sum, self.N), 1)
        return average

    def pre_process_speed_plot(self, save_path):

        x = self.x
        y = self.pre_process_speed
        plt.figure(num = 1)
        plt.plot(x, y)
        plt.grid(visible = True, linewidth = 0.5)
        plt.title('Pre-Process Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N - 1)
        plt.ylim(0, max(y) + 1)
        plt.savefig(str(save_path) +'/Pre-Process Speed')

    def inference_average_speed(self):
        # Compute average inference speed in milliseconds (ms)
        sum = np.sum(self.inference_speed)
        average = round(np.divide(sum, self.N), 1)
        return average

    def inference_speed_plot(self, save_path):

        x = self.x
        y = self.inference_speed
        plt.figure(num = 2)
        plt.plot(x, y)
        plt.grid(visible = True, linewidth = 0.5)
        plt.title('Inference Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N - 1)
        plt.ylim(0, max(y) + 300)
        plt.savefig(str(save_path) + '/Inference Speed')

    def post_process_average_speed(self):
        # Compute average post-process speed in milliseconds (ms)
        sum = np.sum(self.post_process_speed)
        average = round(np.divide(sum, self.N), 1)
        return average

    def post_process_speed_plot(self, save_path):

        x = self.x
        y = self.post_process_speed
        plt.figure(num = 3)
        plt.plot(x, y)
        plt.grid(visible = True, linewidth = 0.5)
        plt.title('Post-Process Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N - 1)
        plt.ylim(0, max(y) + 3)
        plt.savefig(str(save_path) + '/Post-Process Speed')

    def man_down_average_speed(self):
        # Compute average man down classifier speed in milliseconds (ms)
        sum = np.sum(self.man_down_speed)
        average = round(np.divide(sum, self.N), 1)
        return average

    def man_down_speed_plot(self, save_path):
        
        x = self.x
        y = self.man_down_speed
        plt.figure(num = 4)
        plt.plot(x, y)
        plt.grid(visible = True, linewidth = 0.5)
        plt.title('Man Down Classifier Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N - 1)
        plt.ylim(0, max(y) + 0.3)
        plt.savefig(str(save_path) + '/Man Down Classifier Speed')

    def deep_sort_average_speed(self):
        # Compute average DeepSORT speed in milliseconds (ms)
        sum = np.sum(self.deep_sort_speed)
        average = round(np.divide(sum, self.N), 1)
        return average

    def deep_sort_speed_plot(self, save_path):
        
        x = self.x
        y = self.deep_sort_speed
        plt.figure(num = 5)
        plt.plot(x, y)
        plt.grid(visible = True, linewidth = 0.5)
        plt.title('DeepSORT Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N - 1)
        plt.ylim(0, max(y) + 20)
        plt.savefig(str(save_path) + '/DeepSORT Speed')

    def algorithm_average_speed(self):
        # Compute average algorithm speed in milliseconds (ms)
        y = []
        for i in range(0, self.N):
            temp = (self.pre_process_speed[i] + self.inference_speed[i] + self.post_process_speed[i] 
                + self.man_down_speed[i] + self.deep_sort_speed[i])
            y = np.append(y, temp)
        sum = np.sum(y)
        average = round(np.divide(sum, self.N), 1)

        return average

    def algorithm_speed_plot(self, save_path):

        x = self.x
        y = []
        for i in range(0, self.N):
            sum = (self.pre_process_speed[i] + self.inference_speed[i] + self.post_process_speed[i] 
                + self.man_down_speed[i] + self.deep_sort_speed[i])
            y = np.append(y, sum)
        plt.figure(num = 6)
        plt.plot(x, y)
        plt.grid(visible = True, linewidth = 0.5)
        plt.title('Algorithm Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N - 1)
        plt.ylim(0, max(y) + 300)
        plt.savefig(str(save_path) + '/Algorithm Speed')

    def speed_plot(self, save_path):

        self.algorithm_speed = []
        for i in range(0, self.N):
            sum = (self.pre_process_speed[i] + self.inference_speed[i] + self.post_process_speed[i] 
                + self.man_down_speed[i] + self.deep_sort_speed[i])
            self.algorithm_speed = np.append(self.algorithm_speed, sum)

        plt.figure(num = 7)
        plt.plot(self.x, self.algorithm_speed, label = 'Total')
        plt.plot(self.x, self.deep_sort_speed, label = 'DeepSORT')
        plt.plot(self.x, self.man_down_speed, label = 'Man Down Classifier')
        plt.plot(self.x, self.post_process_speed, label = 'Post-Process')
        plt.plot(self.x, self.inference_speed, label = 'YOLO Inference')
        plt.plot(self.x, self.pre_process_speed, label = 'Pre-Process')
        plt.grid(visible = True, linewidth = 0.5)
        plt.title('Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N - 1)
        plt.ylim(0, max(self.algorithm_speed) + 100)
        # plt.legend()
        plt.savefig(str(save_path) + '/Speed')

class ShowCPU:

    def __init__(self, file_path):

        # Define variables:
        self.CPU_utilization_rate = []
        self.CPU_temperature = []
        self.CPU_power_consumption = []

        # Read data from file:
        with open(str(file_path + '/info.txt'), 'r') as f:
            data = f.readlines()
            self.N = len(data)

        for s in data:
            s = s.replace('\n', '')
            s = s.split(' ')
            self.CPU_utilization_rate.append(round(float(s[5]), 1))
            self.CPU_temperature.append(round(float(s[6]), 1))
            self.CPU_power_consumption.append(round(float(s[7]), 1))

        # Define number of frames:
        self.x = []
        for i in range(0, self.N):
            self.x.append(i)

    def CPU_average_utilization_rate(self):
        # Compute average CPU utilization rate in %
        sum = np.sum(self.CPU_utilization_rate)
        average = round(np.divide(sum, self.N), 1)
        return average

    def CPU_utilization_rate_plot(self, save_path):
        
        x = self.x
        y = self.CPU_utilization_rate
        plt.figure(num = 8)
        plt.plot(x, y)
        plt.grid(visible = True, linewidth = 0.5)
        plt.title('CPU Utilization Rate')
        plt.xlabel('Frame [N]')
        plt.ylabel('Utilization Rate [%]')
        plt.xlim(0, self.N - 1)
        plt.ylim(0, max(y) + 20)
        plt.savefig(str(save_path) + '/CPU Utilization Rate')

    def CPU_average_temperature(self):
        # Compute average CPU temperature in Celsius degrees (°C)
        sum = np.sum(self.CPU_temperature)
        average = round(np.divide(sum, self.N), 1)
        return average

    def CPU_temperature_plot(self, save_path):
        
        x = self.x
        y = self.CPU_temperature
        plt.figure(num = 9)
        plt.plot(x, y)
        plt.grid(visible = True, linewidth = 0.5)
        plt.title('CPU Temperature')
        plt.xlabel('Frame [N]')
        plt.ylabel('Temperature [°C]')
        plt.xlim(0, self.N - 1)
        plt.ylim(0, max(y) + 40)
        plt.savefig(str(save_path) + '/CPU Temperature')

    def CPU_mean_power_consumption(self):
        average = []
        for i in range(0, self.N):
            sum = np.sum(self.CPU_power_consumption[0:i+1])
            average.append(round(np.divide(sum, i+1), 1))
        return average

    def CPU_average_power_consumption(self):
        # Compute average GPU utilization rate in %
        sum = np.sum(self.CPU_power_consumption)
        average = round(np.divide(sum, self.N), 1)
        return average

    def CPU_power_consumption_plot(self, save_path):

        x = self.x
        y = self.CPU_power_consumption
        mean = np.round(np.sum(self.CPU_power_consumption)/self.N, 1)
        plt.figure(num = 10)
        plt.plot(x, y)
        plt.plot(x, self.CPU_mean_power_consumption(), label = f"""mean ({mean} W)""")
        plt.grid(visible = True, linewidth = 0.5)
        plt.title('CPU Power Consumption')
        plt.xlabel('Frame [N]')
        plt.ylabel('Power Consumption [W]')
        plt.xlim(0, self.N - 1)
        plt.ylim(0, max(y) + 20)
        plt.legend()
        plt.savefig(str(save_path) + '/CPU Power Consumption')

class ShowGPU:

    def __init__(self, file_path):

        # Define variables:
        self.GPU_memory_used = []
        self.GPU_utilization_rate = []
        self.GPU_temperature = []
        self.GPU_power_consumption = []

        # Read data from file:
        with open(str(file_path + '/info.txt'), 'r') as f:
            data = f.readlines()
            self.N = len(data)

        for s in data:
            s = s.replace('\n', '')
            s = s.split(' ')
            self.GPU_memory_used.append(float(s[8]))
            self.GPU_utilization_rate.append(float(s[9]))
            self.GPU_temperature.append(float(s[10]))
            self.GPU_power_consumption.append(float(s[11]))

        # Define number of frames:
        self.x = []
        for i in range(0, self.N):
            self.x.append(i)

    def GPU_average_utilization_rate(self):
        # Compute average GPU utilization rate in %
        sum = np.sum(self.GPU_utilization_rate)
        average = round(np.divide(sum, self.N), 1)
        return average
    
    def GPU_utilization_rate_plot(self, save_path):
        
        x = self.x
        y = self.GPU_utilization_rate
        plt.figure(num = 11)
        plt.plot(x, y)
        plt.grid(visible = True, linewidth = 0.5)
        plt.title('GPU Utilization Rate')
        plt.xlabel('Frame [N]')
        plt.ylabel('Utilization Rate [%]')
        plt.xlim(0, self.N - 1)
        plt.ylim(0, max(y) + 15)
        plt.savefig(str(save_path) + '/GPU Utilization Rate')

    def GPU_average_temperature(self):
        # Compute average GPU temperature in Celsius degrees (°C)
        sum = np.sum(self.GPU_temperature)
        average = round(np.divide(sum, self.N), 1)
        return average

    def GPU_temperature_plot(self, save_path):
        
        x = self.x
        y = self.GPU_temperature
        plt.figure(num = 12)
        plt.plot(x, y)
        plt.grid(visible = True, linewidth = 0.5)
        plt.title('GPU Temperature')
        plt.xlabel('Frame [N]')
        plt.ylabel('Temperature [°C]')
        plt.xlim(0, self.N - 1)
        plt.ylim(0, max(y) + 30)
        plt.savefig(str(save_path) + '/GPU Temperature')

    def GPU_mean_power_consumption(self):
        average = []
        for i in range(0, self.N):
            sum = np.sum(self.GPU_power_consumption[0:i+1])
            average.append(round(np.divide(sum, i+1), 1))
        return average

    def GPU_average_power_consumption(self):
        # Compute average GPU utilization rate in %
        sum = np.sum(self.GPU_power_consumption)
        average = round(np.divide(sum, self.N), 1)
        return average

    def GPU_power_consumption_plot(self, save_path):

        x = self.x
        y = self.GPU_power_consumption
        mean = np.round(np.sum(self.GPU_power_consumption)/self.N, 1)
        plt.figure(num = 13)
        plt.plot(x, y)
        plt.plot(x, self.GPU_mean_power_consumption(), label = f"""mean ({mean} W)""")
        plt.grid(visible = True, linewidth = 0.5)
        plt.title('GPU Power Consumption')
        plt.xlabel('Frame [N]')
        plt.ylabel('Power Consumption [W]')
        plt.xlim(0, self.N - 1)
        plt.ylim(0, max(y) + 25)
        plt.legend()
        plt.savefig(str(save_path) + '/GPU Power Consumption')