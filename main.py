import os
from src.train_model import main as train_main

def check_requirements():
    """Check if dataset exists"""
    if not os.path.exists("data/topic_dataset.csv"):
        print("Error: Dataset not found. Please create 'topic_dataset.csv' in the data folder.")
        return False
    return True

def main():
    while True:
        print("\n=== Topic Analysis System ===")
        print("1. Train Model")
        print("2. Run GUI Application")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            if check_requirements():
                print("\nStarting model training...")
                train_main()
                print("\nModel training completed!")
            else:
                print("Please generate the dataset first.")
        elif choice == "2":
            if not os.path.exists("models/trained_model.pkl"):
                print("Trained model not found. Please train the model first.")
                continue
            try:
                from gui import main as gui_main
                print("Launching GUI application...")
                gui_main()
            except ImportError as e:
                print(f"Error importing GUI: {e}")
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
