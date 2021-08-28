import os

if __name__ == "__main__":
    for folder in ["train", "valid"]:
        label_dir = f"data2/{folder}/labels"
        files = os.listdir(label_dir)
        with open(f"data2/_annotations_{folder}.txt", "w") as annotation:
            for file in files:
                with open(f'{label_dir}/{file}', "r") as f:
                    annotation.write(f'{file[:-4]}.jpg {",".join(f.read().split(" "))}\n')