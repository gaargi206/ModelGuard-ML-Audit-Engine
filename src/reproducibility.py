import platform
import sys
import pandas
import numpy
import sklearn


def get_reproducibility_info(
    dataset_name,
    target_column,
    random_state=42
):
    """
    Collect environment and configuration
    information required to reproduce an audit.
    """

    reproducibility_info = [

        {
            "Check": "Python Version",
            "Value": platform.python_version(),
            "Status": "Recorded"
        },

        {
            "Check": "Python Executable",
            "Value": sys.executable,
            "Status": "Recorded"
        },

        {
            "Check": "Operating System",
            "Value": platform.system(),
            "Status": "Recorded"
        },

        {
            "Check": "OS Version",
            "Value": platform.platform(),
            "Status": "Recorded"
        },

        {
            "Check": "Scikit-learn Version",
            "Value": sklearn.__version__,
            "Status": "Recorded"
        },

        {
            "Check": "Pandas Version",
            "Value": pandas.__version__,
            "Status": "Recorded"
        },

        {
            "Check": "NumPy Version",
            "Value": numpy.__version__,
            "Status": "Recorded"
        },

        {
            "Check": "Random Seed",
            "Value": str(random_state),
            "Status": "Configured"
        },

        {
            "Check": "Dataset",
            "Value": dataset_name,
            "Status": "Recorded"
        },

        {
            "Check": "Target Variable",
            "Value": target_column,
            "Status": "Recorded"
        }
    ]

    return reproducibility_info


def get_model_configuration(
    random_state=42,
    test_size=0.20
):
    """
    Return the configuration used by the
    baseline model.
    """

    return {
        "Random Seed": random_state,
        "Test Size": test_size,
        "Model": "Logistic Regression",
        "Categorical Encoding": "OneHotEncoder",
        "Numeric Imputation": "Median",
        "Categorical Imputation": "Most Frequent"
    }