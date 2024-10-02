import os

def get_datasets() -> tuple[int, list[tuple[int, str]]]:
    directory = os.getcwd() + "/datasets/"
    n = 0
    fileData = []

    for (root, dirs, files) in os.walk(directory):
        for file in files:
            fileData.append((n, f"{root}/{file}"))
            tokens = file.split("_")
            num = ""
            for element in tokens:
                if element[0] == "n":
                    num = element[1:]
                    break
            n += int(num)

    return (n, fileData)
