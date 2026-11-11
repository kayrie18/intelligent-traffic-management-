"""Regenerate everything: data -> model -> simulation -> charts and website data."""
from src import generate_data, train_model, run_simulation

if __name__ == "__main__":
    generate_data.main()
    train_model.main()
    run_simulation.main()
