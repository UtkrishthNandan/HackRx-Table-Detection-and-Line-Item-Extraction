import os
import sys

def process_folder(folder_path):
    os.system("py app.py")

def main():   
    if len(sys.argv) != 2:
        print("Usage: python main.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    process_folder(folder_path)


if __name__ == "__main__":
    main()