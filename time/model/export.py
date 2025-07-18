import os

from pipeline.export import export_model
from model import SimpleTimeNetwork


def main():
    checkpoint_path = f"{os.getcwd()}\\checkpoints"
    output_path = f"{os.getcwd()}\\..\\..\\weights\\time"
    export_model(SimpleTimeNetwork, checkpoint_path, output_path)


if __name__ == "__main__":
    main()
