from dataset.sst2.load_sst2 import load_train_dataset


def main():
    train_dataset = load_train_dataset()
    print(train_dataset[0:5])
    print(train_dataset.size)


if __name__ == '__main__':
    main()

