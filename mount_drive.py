from google.colab import drive

def mount_drive():
    drive.mount('/content/drive')

if __name__ == '__main__':
    mount_drive()