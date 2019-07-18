import reader
import sys
import os

if sys.argv[1] == "trim":
    gt = reader.getVideoList(data_path="./hw4_data/TrimmedVideos/label/gt_valid.csv")["Action_labels"]
    total = len(gt)

    p = ["", "./p1_valid.txt", "./p2_result.txt"]

    cnt, i = 0, 0
    with open(p[int(sys.argv[2])], "r") as f:
        for line in f:
            if int(gt[i]) == int(line):
                cnt += 1
            i += 1

    print(cnt/total)

else:
    gt_path = "./hw4_data/FullLengthVideos/labels/valid"
    
    file_name = sorted(os.listdir(gt_path))
    tcnt = 0
    ttotal = 0
    for i in range(len(file_name)):
        print(file_name[i])

        gt = []
        with open(os.path.join(gt_path, file_name[i]), "r") as f:
            for line in f:
                line = line.strip()
                gt.append(int(line))
        pre = []
        with open(os.path.join(".", file_name[i]), "r") as f:
            for line in f:
                line = line.strip()
                pre.append(int(line))
        
        total = len(gt)
        cnt = 0
        for j in range(total):
            if gt[j] == pre[j]:
                cnt += 1
        
        print("{}, {}/{}".format(cnt/total, cnt, total))
        tcnt += cnt
        ttotal += total
    
    print("AVG={}, {}/{}".format(tcnt/ttotal, tcnt, ttotal))