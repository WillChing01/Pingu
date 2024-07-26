import numpy as np
import os

def main():
    return 0

    # print("Shuffling dataset...")

    # indices = np.array([i for i in range(n)])
    # np.random.shuffle(indices)

    # validation_num = int(round(VALIDATION_RATIO * n))
    # training_num = n - validation_num

    # print("Generating", validation_num, "validation samples...")

    # validation_input = np.memmap("validation_input_"+str(validation_num)+"_32.dat", mode = "w+", dtype = np.short, shape = (validation_num, 32))
    # validation_labels = np.memmap("validation_labels_"+str(validation_num)+"_1.dat", mode = "w+", dtype = np.short, shape = (validation_num, 1))
    
    # for i in range(0, validation_num):
    #     validation_input[i] = np.array(sparse_input[indices[i]], copy=True)
    #     validation_labels[i][0] = labels[indices[i]][0]
    # validation_input.flush()
    # validation_labels.flush()

    # print("Generating", n - validation_num, "training samples...")

    # training_input = np.memmap("training_input_"+str(n - validation_num)+"_32.dat", mode = "w+", dtype = np.short, shape = (n - validation_num, 32))
    # training_labels = np.memmap("training_labels_"+str(n - validation_num)+"_1.dat", mode = "w+", dtype = np.short, shape = (n - validation_num, 1))

    # for i in range(validation_num, n):
    #     if (i - validation_num) % 1000000 == 0:
    #         print("Progress", i - validation_num)
    #     training_input[i-validation_num] = np.array(sparse_input[indices[i]], copy=True)
    #     training_labels[i-validation_num][0] = labels[indices[i]][0]
    # training_input.flush()
    # training_labels.flush()

if __name__ == "__main__":
    main()
