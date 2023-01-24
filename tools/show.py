
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class Plots:

    def __init__(self, file_path, device):

        # Define variables:
        self.pre_process_speed = []
        self.inference_speed = []
        self.post_process_speed = []
        self.man_down_speed = []
        self.deep_sort_speed = []
        self.GPU_memory_used = []
        self.GPU_utilization_rate = []
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
            self.post_process_speed.append(round(float(s[2]), 1))
            self.man_down_speed.append(round(float(s[3]), 1))
            self.deep_sort_speed.append(round(float(s[4]), 1))
            if device != 'cpu':
                self.GPU_memory_used.append(s[5])
                self.GPU_utilization_rate.append(s[6])
                self.GPU_temperature.append(float(s[7]))
                self.GPU_power_consumption.append(float(s[8]))

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

        plt.plot(x, y)
        plt.grid(visible = True)
        plt.title('Pre-Process Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N)
        plt.ylim()
        plt.savefig('Pre-Process Speed')

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

        plt.plot(x, y)
        plt.grid(visible = True)
        plt.title('Inference Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N)
        plt.ylim(0, max(y) + 40)
        plt.savefig('Inference Speed')

    ### POST-PROCESS SPEED ###

    def post_process_average_speed(self):
        # Compute average post-process speed in milliseconds (ms)
        N = self.N
        sum = np.sum(self.post_process_speed)
        average = round(np.divide(sum, N), 1)

        return average

    def post_process_speed_plot(self):

        x = self.x
        y = self.post_process_speed
        # y[0] = self.average_post_process_speed()
        # y[1] = y[0]

        plt.plot(x, y)
        plt.grid(visible = True)
        plt.title('Post-Process Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N)
        plt.ylim(0, max(y) + 5)
        plt.savefig('Post-Process Speed')

    ### MANDOWN CLASSIFIER SPEED ###

    def man_down_average_speed(self):
        # Compute average man down classifier speed in milliseconds (ms)
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
        plt.grid(visible = True)
        plt.title('Man Down Classifier Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N)
        plt.ylim(-1, max(y) + 5)
        plt.savefig('Man Down Classifier Speed')

    ### DEEP SORT ALGORITHM SPEED ###

    def deep_sort_average_speed(self):
        # Compute average DeepSORT speed in milliseconds (ms)
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
        plt.grid(visible = True)
        plt.title('DeepSORT Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N)
        plt.ylim(0, max(y) + 10)
        plt.savefig('DeepSORT Speed')

    ### ALGORITHM TOTAL SPEED ###

    def total_average_speed(self):
        # Compute average total speed in milliseconds (ms)
        N = self.N
        y = []
        for i in range(0, self.N):
            temp = (self.pre_process_speed[i] + self.inference_speed[i] + self.post_process_speed[i] 
                + self.man_down_speed[i] + self.deep_sort_speed[i])
            y = np.append(y, temp)
        sum = np.sum(y)
        average = round(np.divide(sum, N), 1)

        return average

    def total_speed_plot(self):

        x = self.x
        y = []
        for i in range(0, self.N):
            sum = (self.pre_process_speed[i] + self.inference_speed[i] + self.post_process_speed[i] 
                + self.man_down_speed[i] + self.deep_sort_speed[i])
            y = np.append(y, sum)

        plt.plot(x, y)
        plt.grid(visible = True)
        plt.title('Total Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N)
        plt.ylim(0, max(y) + 10)
        plt.savefig('Total Speed')

    ### TOTAL ###

    def speed_plot(self):

        self.total_speed = []
        for i in range(0, self.N):
            sum = (self.pre_process_speed[i] + self.inference_speed[i] + self.post_process_speed[i] 
                + self.man_down_speed[i] + self.deep_sort_speed[i])
            self.total_speed = np.append(self.total_speed, sum)

        plt.figure(num = 1)
        plt.plot(self.x, self.total_speed, label = 'Total')
        plt.plot(self.x, self.deep_sort_speed, label = 'DeepSORT')
        plt.plot(self.x, self.man_down_speed, label = 'Man Down Classifier')
        plt.plot(self.x, self.post_process_speed, label = 'Post-Process')
        plt.plot(self.x, self.inference_speed, label = 'YOLO Inference')
        plt.plot(self.x, self.pre_process_speed, label = 'Pre-Process')
        plt.grid()
        plt.title('Speed')
        plt.xlabel('Frame [N]')
        plt.ylabel('Time [ms]')
        plt.xlim(0, self.N)
        plt.ylim(-5, max(self.total_speed) + 10)
        # plt.legend()
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