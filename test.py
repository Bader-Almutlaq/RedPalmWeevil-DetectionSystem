with open("./data/mixed_dataset/metadata.txt", "r") as file:
    lines = file.readlines()
    new_lines = [lines[0]]
    for i, line in enumerate(lines[1:], 1):
        col = line.split(",")
        col[0] = str(i)
        new_lines.append(",".join(col))
with open("./data/meatdata.txt", "w") as file:
    file.writelines(new_lines)
