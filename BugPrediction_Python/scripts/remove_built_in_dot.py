import os
from util.Config import Config
import re
from tqdm import tqdm

class RmBiDot:
    def __init__(self):
        self.dot_path = Config.dot_path

    def remove(self):
        print("start remove")
        delete_files = []
        for root, dirs, files in os.walk(self.dot_path):
            for file in files:
                if file.find(".dot") > -1:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as r:
                        content = r.read();

                    pattern = re.compile("\<SUB\>")
                    if pattern.search(content) is None and file.find(".dot") > -1:
                        delete_files.append(os.path.join(root, file))

        with tqdm(total=len(delete_files), desc='(T)') as pbar:
            for df in delete_files:
                os.remove(df)
                pbar.update()

        print("remove completed")

def main():
    rm = RmBiDot()
    rm.remove()


if __name__ == "__main__":
    main()
