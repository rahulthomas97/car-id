import os
import json
import matplotlib.pyplot as plt
import torchvision.datasets as datasets

def plot_class_distribution(data_dir, save_path):
    # Check if the class distribution file exists
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            class_counts = json.load(f)
        print("Loaded class distribution from file.")
    else:
        dataset = datasets.StanfordCars(root=data_dir, split="train", download=False)
        class_counts = {}

        for _, idx in dataset:
            label = dataset.classes[idx]
            class_counts[label] = class_counts.get(label, 0) + 1

        # Save the class distribution to a file
        with open(save_path, 'w') as f:
            json.dump(class_counts, f)
        print("Saved class distribution to file.")

    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    print("Max number of classes:", max(counts))
    print("Min number of classes:", min(counts))
    print("average number of classes:", sum(counts) / len(counts))
    print("standard deviation of classes:", sum([(x - sum(counts) / len(counts)) ** 2 for x in counts]) / len(counts))

    # plt.figure(figsize=(20, 30))
    # plt.barh(classes, counts, color='skyblue')
    # plt.xlabel('Number of Images')
    # plt.ylabel('Classes')
    # plt.title('Number of Images per Class in Stanford Cars Dataset')
    # plt.show()

if __name__ == "__main__":
    data_dir = "./"  # Change this to your dataset directory if needed
    save_path = "./class_distribution.json"  # Path to save the class distribution
    plot_class_distribution(data_dir, save_path)
