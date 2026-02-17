# The Data Science Quickstart Guide — Companion Code Repository

*By [Chaitanya Krishna Kasaraneni](https://ckasaraneni.com/), [Srikar Kashyap Pulipaka](https://srikarkashyap.com/), Sarmista Thalapaneni*

---

## About This Repository

This repository contains all companion code examples for *The Data Science Quickstart Guide*. Each chapter folder includes runnable Python scripts that demonstrate the key concepts discussed in the book.

## Repository Structure

| Chapter | Topic | Script | Key Libraries |
|---------|-------|--------|---------------|
| 2 | Data Collection & Management | `data_cleaning.py` | pandas |
| 3 | Exploratory Data Analysis | `eda_visualization.py` | pandas, matplotlib |
| 4 | Data Wrangling & Transformation | `data_transformation.py` | pandas |
| 5 | Statistical Foundations | `hypothesis_testing.py` | scipy, numpy |
| 7 | Supervised Learning Techniques | `linear_regression.py` | scikit-learn, pandas |
| 8 | Unsupervised Learning Techniques | `clustering_comparison.py` | scikit-learn, matplotlib |
| 9 | Deep Learning & Neural Networks | `neural_network.py` | tensorflow, keras |
| 10 | Natural Language Processing | `text_processing.py` | nltk |
| 11 | Data Visualization Techniques | `visualization_examples.py` | matplotlib, numpy |
| 14 | Data Science for Business Analytics | `customer_segmentation.py` | scikit-learn, pandas |
| 15 | Time Series Analysis | `arima_forecasting.py` | statsmodels, matplotlib |
| 17 | Data Science Tools & Technologies | `jupyter_workflow.ipynb` | pandas, matplotlib, scikit-learn |
| 18 | The Role of AI in Data Science | `automl_gridsearch.py` | scikit-learn |

## Getting Started

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/chaitanyakasaraneni/data-science-quickstart-code.git
cd data-science-quickstart-code

# Install all dependencies
pip install -r requirements.txt
```

### Running Examples

Each script is self-contained and can be run independently:

```bash
# Run a specific chapter's example
python chapter-07/linear_regression.py

# Or open the Jupyter notebook
jupyter notebook chapter-17/jupyter_workflow.ipynb
```

Some examples generate sample data internally for demonstration purposes. Where external datasets are referenced (e.g., `housing.csv`, `sales_data.csv`), the scripts include synthetic data generation so they run out of the box without additional downloads.

## Chapter Overview

- **Chapters 1, 6, 12, 13, 16, 19, 20** do not have companion code — these chapters cover conceptual, ethical, or forward-looking topics best suited to prose discussion. Diagrams for these chapters are included in the book itself.

## Requirements

All dependencies are listed in `requirements.txt`. The major libraries used are:

- **pandas** — data manipulation and analysis
- **numpy** — numerical computing
- **matplotlib** — data visualization
- **scikit-learn** — machine learning
- **scipy** — statistical testing
- **statsmodels** — time series analysis
- **tensorflow** — deep learning
- **nltk** — natural language processing

## License

This source code is provided as a companion to the book *The Data Science Quickstart Guide*.

## Contact

For questions or errata, please open an issue in this repository.
