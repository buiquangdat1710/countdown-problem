from src.dataset import load_csv_dataset_sft
from src.dataset.sft import map_problem_description_to_conversation_sft


def main() -> None:
    """
    Main function
    """
    dataset = load_csv_dataset_sft(
        "data/sft/train.csv", "train", map_problem_description_to_conversation_sft
    )
    print(dataset)


if __name__ == "__main__":
    main()
