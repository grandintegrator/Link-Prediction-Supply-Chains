# Link Prediction in Supply Chain Networks

Repo containing MPhil thesis code for link prediction in supply chains with a focus on Jaguar Land Rover.

The file structure is laid out as follows:

Start analysis with `pip install -r requirements.txt`

`
├── config              <- Project configurations
│   ├── config.yml      <- For changing run paramteres (e.g. number of epochs 🌝)
├── data
│   ├── 01_raw          <- Data from third party sources.
│   ├── 02_intermediate <- Intermediate data that has been transformed.
│   ├── 03_models       <- Saved GNN models
│   └── 04_results      <- Results from the analysis 🚀
│
├── README.md           <- The top-level README for developers using this project.
├── docs                <- A default Sphinx project; see sphinx-doc.org for details
├── src
│   ├── exploration     <- Exploring the underlying data (e.g. degree distributions)
│   ├── ingestion       <- Transforming the complex network into a Knowledge Graph and build Pytorch dataloaders
│   ├── managers        <- Training and testing managers for pytorch (`training.py` and `testing.py`)
│   └── model           <- DGL Models
`
