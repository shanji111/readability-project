from readability.train import train

if __name__ == "__main__":
    train(csv_path="data/converted_all_data.csv",
          output_dir="../model",
          num_epochs=3,
          batch_size=8)
