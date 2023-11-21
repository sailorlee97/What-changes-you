import threading
import psutil
import GPUtil
import time

class CpuGpuMonitor:
    def __init__(self):
        self.cpu_data = []
        self.gpu_data = []
        self.flag = threading.Event()
        self.monitor_thread = None

    def start(self):
        self.flag.set()
        self.monitor_thread = threading.Thread(target=self.update_usage)
        self.monitor_thread.start()

    def end(self):
        # Set the flag to stop the monitoring thread
        self.flag.clear()
        # Wait for the monitoring thread to finish
        self.monitor_thread.join()
        # Print the average CPU and GPU usage
        print(f"Average CPU usage: {self.get_cpu_average_usage()}%")
        print(f"Average GPU usage: {self.get_gpu_average_usage()}%")

    def update_usage(self):
        while self.flag.is_set():
            time.sleep(1)
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_data.append(cpu_percent)
            gpu_percent = GPUtil.getGPUs()[0].load * 100
            self.gpu_data.append(gpu_percent)

    def get_cpu_average_usage(self):
        return sum(self.cpu_data) / len(self.cpu_data) if len(self.cpu_data) > 0 else 0

    def get_gpu_average_usage(self):
        return sum(self.gpu_data) / len(self.gpu_data) if len(self.gpu_data) > 0 else 0

if __name__ == '__main__':

    # 创建监控对象
    monitor = CpuGpuMonitor()

    # 开始监控
    monitor.start()

    # 持续更新CPU和GPU使用率数据
    while True:
        # 更新CPU和GPU使用率数据
        monitor.update_usage()

    # 结束监控
    monitor.end()

    # 输出CPU和GPU平均使用率
    print(f"CPU average usage: {monitor.get_cpu_average_usage():.2f}%")
    print(f"GPU average usage: {monitor.get_gpu_average_usage():.2f}%")
