"""
This file is used to test the performance of the tornado server.
"""
import requests
import time
import random
import os 
import threading 
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

TARGET_IP = 'http://localhost'
TARGET_URL = f'{TARGET_IP}:9263/submit'

punctuations = ['.', '!', '?']
def generate_random_text(length=100):
    text = []
    for i in range(length):
        text.append(chr(random.randint(97, 122)))
        if i % 10 == 0 and i != 0:
            text.append(random.choice(punctuations))
    return ''.join(text)

test_grid = {
    'num_reqs_one_time': [1, 5, 10, 20, 30, 40,],
    'length': [10, 100, 200, 300, 400, 500]
}

def unit_test(num_reqs_one_time=10, length=1000, timeout=5):
    """
    Send multiple requests to the server at the same time.

    Calculate the average time of the requests.

    :param num_reqs_one_time: the number of requests sent at the same time.
    :param length: the length of the text.

    :return: the average time of the requests.
    """
    text = generate_random_text(length)
    is_timeout = False
    start_time = time.time()

    def send_request():
        try:
            response = requests.post(TARGET_URL, data={'data': text}, timeout=timeout)
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
        except requests.exceptions.Timeout:
            nonlocal is_timeout
            is_timeout = True
            print(f"Error: Request timed out after {timeout} seconds")
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")

    threads = []
    for _ in range(num_reqs_one_time):
        thread = threading.Thread(target=send_request)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.time()
    if is_timeout:
        print("Average time per request: Timeout")
        return -1
    average_time = (end_time - start_time) / num_reqs_one_time
    print(f"Average time per request: {average_time:.4f} seconds")
    return average_time

def main():
    x_axis = []
    y_axis = []
    z_axis = []
    for num_reqs_one_time in test_grid['num_reqs_one_time']:
        for length in test_grid['length']:
            print(f"Number of requests: {num_reqs_one_time}, Length of text: {length}")
            avg_time = unit_test(num_reqs_one_time, length)
            if avg_time != -1:
                x_axis.append(num_reqs_one_time)
                y_axis.append(length)
                z_axis.append(avg_time)

    x_axis = np.array(x_axis).reshape((len(test_grid['num_reqs_one_time']), len(test_grid['length'])))
    y_axis = np.array(y_axis).reshape((len(test_grid['num_reqs_one_time']), len(test_grid['length'])))
    z_axis = np.array(z_axis).reshape((len(test_grid['num_reqs_one_time']), len(test_grid['length'])))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_axis, y_axis, z_axis, cmap='viridis')
    ax.set_xlabel('Number of requests')
    ax.set_ylabel('Length of text')
    ax.set_zlabel('Average time per request')
    plt.show()
            

if __name__ == '__main__':
    main()